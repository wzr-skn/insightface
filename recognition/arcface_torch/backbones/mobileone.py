# Copyright (c) OpenMMLab. All rights reserved.
# Modified from official impl https://github.com/apple/ml-mobileone/blob/main/mobileone.py  # noqa: E501
from typing import Optional, Sequence

import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, Sequential, Module, ModuleList
import torch
import torch.nn.functional as F

from torch.nn.modules.batchnorm import _BatchNorm

from .mobilefacenet import ConvBlock, GDC


class MobileOneBlock(Module):
    """MobileOne block for MobileOne backbone.

    Args:
        in_channels (int): The input channels of the block.
        out_channels (int): The output channels of the block.
        kernel_size (int): The kernel size of the convs in the block. If the
            kernel size is large than 1, there will be a ``branch_scale`` in
             the block.
        num_convs (int): Number of the convolution branches in the block.
        stride (int): Stride of convolution layers. Defaults to 1.
        padding (int): Padding of the convolution layers. Defaults to 1.
        dilation (int): Dilation of the convolution layers. Defaults to 1.
        groups (int): Groups of the convolution layers. Defaults to 1.
        se_cfg (None or dict): The configuration of the se module.
            Defaults to None.
        act_cfg (dict): Config dict for activation layer.
            Defaults to ``dict(type='ReLU')``.
        deploy (bool): Whether the model structure is in the deployment mode.
            Defaults to False.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 num_convs: int,
                 stride: int = 1,
                 padding: int = 1,
                 dilation: int = 1,
                 groups: int = 1,
                 se_cfg: Optional[dict] = None,
                 act_cfg: Optional[dict] = dict(type='ReLU'),
                 deploy: bool = False,
                 act: bool = True):
        super().__init__()

        assert se_cfg is None or isinstance(se_cfg, dict)
        if se_cfg is not None:
            raise NotImplementedError("se module is not yet implemented")
        else:
            self.se = nn.Identity()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_conv_branches = num_convs
        self.stride = stride
        self.padding = padding
        self.se_cfg = se_cfg
        self.act_cfg = act_cfg
        self.deploy = deploy
        self.groups = groups
        self.dilation = dilation

        if deploy:
            self.branch_reparam = Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                groups=self.groups,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=True)
        else:
            # judge if input shape and output shape are the same.
            # If true, add a normalized identity shortcut.
            if out_channels == in_channels and stride == 1:
                self.branch_norm = BatchNorm2d(in_channels)
            else:
                self.branch_norm = None

            self.branch_scale = None
            if kernel_size > 1:
                self.branch_scale = self.create_conv_bn(kernel_size=1)

            self.branch_conv_list = ModuleList()
            for _ in range(num_convs):
                self.branch_conv_list.append(
                    self.create_conv_bn(
                        kernel_size=kernel_size,
                        padding=padding,
                        dilation=dilation))

        self.act = PReLU(num_parameters=self.out_channels) if act else None

    def create_conv_bn(self, kernel_size, dilation=1, padding=0):
        """cearte a (conv + bn) Sequential layer."""
        conv_bn = Sequential()
        conv_bn.add_module(
            'conv',
            Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                groups=self.groups,
                stride=self.stride,
                dilation=dilation,
                padding=padding,
                bias=False))
        conv_bn.add_module(
            'norm',
            BatchNorm2d(num_features=self.out_channels))

        return conv_bn

    def forward(self, x):

        def _inner_forward(inputs):
            if self.deploy:
                return self.branch_reparam(inputs)

            inner_out = 0
            if self.branch_norm is not None:
                inner_out = self.branch_norm(inputs)

            if self.branch_scale is not None:
                inner_out += self.branch_scale(inputs)

            for branch_conv in self.branch_conv_list:
                inner_out += branch_conv(inputs)

            return inner_out

        return self.se(_inner_forward(x)) if self.act is None else self.act(self.se(_inner_forward(x)))

    def switch_to_deploy(self):
        """Switch the model structure from training mode to deployment mode."""
        if self.deploy:
            return

        reparam_weight, reparam_bias = self.reparameterize()
        self.branch_reparam = Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=True)
        self.branch_reparam.weight.data = reparam_weight
        self.branch_reparam.bias.data = reparam_bias

        for param in self.parameters():
            param.detach_()
        delattr(self, 'branch_conv_list')
        if hasattr(self, 'branch_scale'):
            delattr(self, 'branch_scale')
        delattr(self, 'branch_norm')

        self.deploy = True

    def reparameterize(self):
        """Fuse all the parameters of all branches.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Parameters after fusion of all
                branches. the first element is the weights and the second is
                the bias.
        """
        weight_conv, bias_conv = 0, 0
        for branch_conv in self.branch_conv_list:
            weight, bias = self._fuse_conv_bn(branch_conv)
            weight_conv += weight
            bias_conv += bias

        weight_scale, bias_scale = 0, 0
        if self.branch_scale is not None:
            weight_scale, bias_scale = self._fuse_conv_bn(self.branch_scale)
            # Pad scale branch kernel to match conv branch kernel size.
            pad = self.kernel_size // 2
            weight_scale = F.pad(weight_scale, [pad, pad, pad, pad])

        weight_norm, bias_norm = 0, 0
        if self.branch_norm:
            tmp_conv_bn = self._norm_to_conv(self.branch_norm)
            weight_norm, bias_norm = self._fuse_conv_bn(tmp_conv_bn)

        return (weight_conv + weight_scale + weight_norm,
                bias_conv + bias_scale + bias_norm)

    def _fuse_conv_bn(self, branch):
        """Fuse the parameters in a branch with a conv and bn.

        Args:
            branch (mmcv.runner.Sequential): A branch with conv and bn.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The parameters obtained after
                fusing the parameters of conv and bn in one branch.
                The first element is the weight and the second is the bias.
        """
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps

        std = (running_var + eps).sqrt()
        fused_weight = (gamma / std).reshape(-1, 1, 1, 1) * kernel
        fused_bias = beta - running_mean * gamma / std

        return fused_weight, fused_bias

    def _norm_to_conv(self, branch_nrom):
        """Convert a norm layer to a conv-bn sequence towards
        ``self.kernel_size``.

        Args:
            branch (nn.BatchNorm2d): A branch only with bn in the block.

        Returns:
            (mmcv.runner.Sequential): a sequential with conv and bn.
        """
        input_dim = self.in_channels // self.groups
        conv_weight = torch.zeros(
            (self.in_channels, input_dim, self.kernel_size, self.kernel_size),
            dtype=branch_nrom.weight.dtype)

        for i in range(self.in_channels):
            conv_weight[i, i % input_dim, self.kernel_size // 2,
                        self.kernel_size // 2] = 1
        conv_weight = conv_weight.to(branch_nrom.weight.device)

        tmp_conv = self.create_conv_bn(kernel_size=self.kernel_size)
        tmp_conv.conv.weight.data = conv_weight
        tmp_conv.norm = branch_nrom
        return tmp_conv


class MobileOneBlock_pairs(Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_conv_branches: int = 1,
                 stride: int = 1,
                 pw_act: bool = True):
        super().__init__()
        self.dw = MobileOneBlock(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                num_convs=num_conv_branches,
                stride=stride,
                padding=1,
                groups=in_channels)

        self.pw = MobileOneBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                num_convs=num_conv_branches,
                stride=1,
                padding=0,
                act=pw_act)

    def forward(self, x):
        return self.pw(self.dw(x))


class MobileOne(Module):
    """MobileOne backbone.

    A PyTorch impl of : `An Improved One millisecond Mobile Backbone
    <https://arxiv.org/pdf/2206.04040.pdf>`_

    Args:
        arch (str | dict): MobileOne architecture. If use string, choose
            from 's0', 's1', 's2', 's3' and 's4'. If use dict, it should
            have below keys:

            - num_blocks (Sequence[int]): Number of blocks in each stage.
            - width_factor (Sequence[float]): Width factor in each stage.
            - num_conv_branches (Sequence[int]): Number of conv branches
              in each stage.
            - num_se_blocks (Sequence[int]): Number of SE layers in each
              stage, all the SE layers are placed in the subsequent order
              in each stage.

            Defaults to 's0'.
        in_channels (int): Number of input image channels. Default: 3.
        out_indices (Sequence[int] | int): Output from which stages.
            Defaults to ``(3, )``.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters. Defaults to -1.
        conv_cfg (dict | None): The config dict for conv layers.
            Defaults to None.
        norm_cfg (dict): The config dict for norm layers.
            Defaults to ``dict(type='BN')``.
        act_cfg (dict): Config dict for activation layer.
            Defaults to ``dict(type='ReLU')``.
        deploy (bool): Whether to switch the model structure to deployment
            mode. Defaults to False.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Defaults to False.
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> import torch
        >>> x = torch.rand(1, 3, 224, 224)
        >>> model = MobileOne("s0", out_indices=(0, 1, 2, 3))
        >>> model.eval()
        >>> outputs = model(x)
        >>> for out in outputs:
        ...     print(tuple(out.shape))
        (1, 48, 56, 56)
        (1, 128, 28, 28)
        (1, 256, 14, 14)
        (1, 1024, 7, 7)
    """

    arch_zoo = {
        's0':
        dict(
            num_blocks=[2, 8, 10],
            width_factor=[0.75, 1.0, 1.0],
            num_conv_branches=[4, 4, 4],
            num_se_blocks=[0, 0, 0]),
        's1':
        dict(
            num_blocks=[2, 8, 10],
            width_factor=[1.5, 1.5, 2.0],
            num_conv_branches=[1, 1, 1],
            num_se_blocks=[0, 0, 0]),
        's2':
        dict(
            num_blocks=[2, 8, 10],
            width_factor=[1.5, 2.0, 2.5],
            num_conv_branches=[1, 1, 1],
            num_se_blocks=[0, 0, 0]),
        's2_depth':
        dict(
            num_blocks=[3, 9, 11],
            width_factor=[1.5, 2.0, 2.5],
            num_conv_branches=[1, 1, 1],
            num_se_blocks=[0, 0, 0]),
        's3':
        dict(
            num_blocks=[2, 8, 10],
            width_factor=[2.0, 2.5, 3.0],
            num_conv_branches=[1, 1, 1],
            num_se_blocks=[0, 0, 0]),
        's4':
        dict(
            num_blocks=[2, 8, 10],
            width_factor=[3.0, 3.5, 3.5],
            num_conv_branches=[1, 1, 1],
            num_se_blocks=[0, 0, 0]),
        's5':
        dict(
            num_blocks=[3, 12, 15],
            width_factor=[4.0, 4.0, 4.5],
            num_conv_branches=[1, 1, 1],
            num_se_blocks=[0, 0, 0])
    }

    def __init__(self,
                 arch="s1",
                 in_channels=3,
                 out_indices=(2, ),
                 frozen_stages=-1,
                 act_cfg=dict(type='ReLU'),
                 se_cfg=dict(ratio=16),
                 deploy=False,
                 norm_eval=False,
                 num_features=512,
                 fp16=True,
                 base_channels=[64, 128, 256]):
        super().__init__()

        if isinstance(arch, str):
            assert arch in self.arch_zoo, f'"arch": "{arch}"' \
                f' is not one of the {list(self.arch_zoo.keys())}'
            arch = self.arch_zoo[arch]
        elif not isinstance(arch, dict):
            raise TypeError('Expect "arch" to be either a string '
                            f'or a dict, got {type(arch)}')

        self.arch = arch
        for k, value in self.arch.items():
            assert isinstance(value, list) and len(value) == 3, \
                f'the value of {k} in arch must be list with 4 items.'

        self.in_channels = in_channels
        self.deploy = deploy
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval

        self.se_cfg = se_cfg
        self.act_cfg = act_cfg

        self.fp16 = fp16
        self.num_features = num_features

        channels = min(64,
                       int(base_channels[0] * self.arch['width_factor'][0]))
        self.stage0 = MobileOneBlock(
            self.in_channels,
            channels,
            stride=2,
            kernel_size=3,
            num_convs=1,
            act_cfg=act_cfg,
            deploy=deploy)

        self.in_planes = channels
        self.stages = []
        for i, num_blocks in enumerate(self.arch['num_blocks']):
            planes = int(base_channels[i] * self.arch['width_factor'][i])

            stage = self._make_stage(planes, num_blocks,
                                     arch['num_se_blocks'][i],
                                     arch['num_conv_branches'][i])

            stage_name = f'stage{i + 1}'
            self.add_module(stage_name, stage)
            self.stages.append(stage_name)

        self.conv_sep = ConvBlock(planes, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.features = GDC(self.num_features)
        self._initialize_weights()

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        out_indices = list(out_indices)
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = len(self.stages) + index
            assert 0 <= out_indices[i] <= len(self.stages), \
                f'Invalid out_indices {index}.'
        self.out_indices = out_indices

    def _make_stage(self, planes, num_blocks, num_se, num_conv_branches):
        strides = [2] + [1] * (num_blocks - 1)
        if num_se > num_blocks:
            raise ValueError('Number of SE blocks cannot '
                             'exceed number of layers.')
        blocks = []
        for i in range(num_blocks):
            use_se = False
            if i >= (num_blocks - num_se):
                use_se = True

            blocks.append(
                # Depthwise conv
                MobileOneBlock(
                    in_channels=self.in_planes,
                    out_channels=self.in_planes,
                    kernel_size=3,
                    num_convs=num_conv_branches,
                    stride=strides[i],
                    padding=1,
                    groups=self.in_planes,
                    se_cfg=self.se_cfg if use_se else None,
                    act_cfg=self.act_cfg,
                    deploy=self.deploy))

            blocks.append(
                # Pointwise conv
                MobileOneBlock(
                    in_channels=self.in_planes,
                    out_channels=planes,
                    kernel_size=1,
                    num_convs=num_conv_branches,
                    stride=1,
                    padding=0,
                    se_cfg=self.se_cfg if use_se else None,
                    act_cfg=self.act_cfg,
                    deploy=self.deploy))

            self.in_planes = planes

        return Sequential(*blocks)

    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            x = self.stage0(x)
            outs = []
            for i, stage_name in enumerate(self.stages):
                stage = getattr(self, stage_name)
                x = stage(x)
        x = self.conv_sep(x.float() if self.fp16 else x)
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.stage0.eval()
            for param in self.stage0.parameters():
                param.requires_grad = False
        for i in range(self.frozen_stages):
            stage = getattr(self, f'stage{i+1}')
            stage.eval()
            for param in stage.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        """switch the mobile to train mode or not."""
        super(MobileOne, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def switch_to_deploy(self):
        """switch the model to deploy mode, which has smaller amount of
        parameters and calculations."""
        for m in self.modules():
            if isinstance(m, MobileOneBlock):
                m.switch_to_deploy()
        self.deploy = True

def get_mbo_s1(fp16, num_features):
    return MobileOne(arch="s1", fp16=fp16, num_features=num_features, base_channels=[64, 128, 256])

def get_mbo_s2(fp16, num_features):
    return MobileOne(arch="s2", fp16=fp16, num_features=num_features, base_channels=[64, 128, 256])

def get_mbo_s2_depth(fp16, num_features):
    return MobileOne(arch="s2_depth", fp16=fp16, num_features=num_features, base_channels=[64, 128, 256])

def get_mbo_s3(fp16, num_features):
    return MobileOne(arch="s3", fp16=fp16, num_features=num_features, base_channels=[64, 128, 256])

def get_mbo_s4(fp16, num_features):
    return MobileOne(arch="s4", fp16=fp16, num_features=num_features, base_channels=[64, 128, 256])

def get_mbo_s5(fp16, num_features):
    return MobileOne(arch="s5", fp16=fp16, num_features=num_features, base_channels=[64, 128, 256])
