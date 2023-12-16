# Copyright (c) OpenMMLab. All rights reserved.
# Modified from official impl https://github.com/apple/ml-mobileone/blob/main/mobileone.py  # noqa: E501
from typing import Optional, Sequence

import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, LayerNorm, GroupNorm, PReLU, Sequential, Module, ModuleList
import torch
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from timm.models.layers import trunc_normal_

from .mobileone import MobileOneBlock, MobileOneBlock_pairs
from .efficientformer import Meta3D, Flat
from .mobilefacenet import ConvBlock, GDC


class Mbo_Efformer(Module):
    """MobileOne and EfficientFormer backbone.

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
        's1':
        dict(
            num_blocks=[2, 8, 10, 6],
            width_factor=[2.0, 2.0, 2.0, None],
            framework=['CNN', 'CNN', 'CNN', 'Transformer'],
            num_conv_branches=[1, 1, 1, None],
            num_se_blocks=[0, 0, 0, None]),
        's3':
        dict(
            num_blocks=[2, 8, 10, 6],
            width_factor=[2.0, 2.5, 3.0, None],
            framework=['CNN', 'CNN', 'CNN', 'Transformer'],
            num_conv_branches=[1, 1, 1, None],
            num_se_blocks=[0, 0, 0, None]),
    }

    def __init__(self,
                 arch="s1",
                 in_channels=3,
                 mlp_ratio=2,
                 reshape_last_feat=False,
                 out_indices=(-1, ),
                 frozen_stages=-1,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 use_layer_scale=True,
                 act_cfg=dict(type='ReLU'),
                 se_cfg=dict(ratio=16),
                 deploy=False,
                 norm_eval=False,
                 num_features=512,
                 fp16=True,
                 base_channels=[64, 128, 256, 512]):
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
            assert isinstance(value, list) and len(value) == 4, \
                f'the value of {k} in arch must be list with 4 items.'

        self.in_channels = in_channels
        self.deploy = deploy
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.out_indices = out_indices
        self.reshape_last_feat = reshape_last_feat

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
            if self.arch['framework'][i] == 'CNN':
                planes = int(base_channels[i] * self.arch['width_factor'][i])
                stage = self._make_cnn_stage(planes, num_blocks,
                                         arch['num_se_blocks'][i],
                                         arch['num_conv_branches'][i])
            elif self.arch['framework'][i] == 'Transformer':
                stage = self._make_transformer_stage(
                            planes,
                            self.arch['num_blocks'],
                            num_blocks,
                            mlp_ratio=mlp_ratio,
                            drop_rate=drop_rate,
                            drop_path_rate=drop_path_rate,
                            use_layer_scale=use_layer_scale)

            stage_name = f'stage{i + 1}'
            self.add_module(stage_name, stage)
            self.stages.append(stage_name)

        for i_layer in self.out_indices:
            if not self.reshape_last_feat and self.arch['num_blocks'][-1] > 0:
                layer = LayerNorm(self.in_planes)
            else:
                # use GN with 1 group as channel-first LN2D
                layer = GroupNorm(num_groups=1, num_channels=self.in_planes)
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.linear = Linear(self.in_planes, self.num_features, bias=False)
        self.bn = BatchNorm1d(self.num_features)

        self.frozen_stages = frozen_stages
        self._freeze_stages()
        self.apply(self._init_weights)

    def _make_cnn_stage(self, planes, num_blocks, num_se, num_conv_branches):
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

    def _make_transformer_stage(self,
                                planes,
                                layers,
                                num_blocks,
                                mlp_ratio=4.,
                                drop_rate=.0,
                                drop_path_rate=0.,
                                use_layer_scale=True,):
        blocks = []
        blocks.append(Flat())

        for block_idx in range(num_blocks):
            block_dpr = drop_path_rate * (block_idx + sum(layers[:-1])) / (
                    sum(layers) - 1)
            blocks.append(
                Meta3D(
                    planes,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=block_dpr,
                    use_layer_scale=use_layer_scale,
                ))

        return Sequential(*blocks)

    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            x = self.stage0(x)
            for i, stage_name in enumerate(self.stages):
                stage = getattr(self, stage_name)
                if self.arch['framework'][i] != 'CNN':
                    break
                x = stage(x)
        x = stage(x.float() if self.fp16 else x)
        for idx in self.out_indices:
            norm_layer = getattr(self, f'norm{idx}')
            x = norm_layer(x).permute((0, 2, 1))
        x = self.gap(x).view(x.size(0), -1)
        x_out = self.bn(self.linear(x))
        return x_out.contiguous()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.stage0.eval()
            for param in self.stage0.parameters():
                param.requires_grad = False

        for i in range(self.frozen_stages):
            # Include both block and downsample layer.
            module = self.network[i]
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                norm_layer.eval()
                for param in norm_layer.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        """switch the mobile to train mode or not."""
        super(Mbo_Efformer, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

def get_mbo_efformer_s1(fp16, num_features):
    return Mbo_Efformer(arch="s1", fp16=fp16, num_features=num_features, mlp_ratio=2,)

def get_mbo_efformer_s3(fp16, num_features):
    return Mbo_Efformer(arch="s3", fp16=fp16, num_features=num_features, mlp_ratio=2,)
