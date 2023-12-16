# Copyright (c) OpenMMLab. All rights reserved.
import itertools

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, LayerNorm, GroupNorm, PReLU, Hardswish, Sequential, Module, ModuleList

from .efficientformer import *

class HybridBackbone(Module):

    def __init__(
            self,
            embed_dim,
            kernel_size=3,
            stride=2,
            pad=1,
            dilation=1,
            groups=1,
    ):
        super(HybridBackbone, self).__init__()

        self.input_channels = [
            3, embed_dim // 4, embed_dim // 2
        ]
        self.output_channels = [
            embed_dim // 4, embed_dim // 2, embed_dim
        ]

        self.patch_embed = Sequential()

        for i in range(len(self.input_channels)):
            conv_bn = MobileOneBlock(
                self.input_channels[i],
                self.output_channels[i],
                stride=stride,
                kernel_size=kernel_size,
                num_convs=4,
                act=False)
            self.patch_embed.add_module('%d' % (2 * i), conv_bn)
            if i < len(self.input_channels) - 1:
                self.patch_embed.add_module('%d' % (i * 2 + 1), Hardswish())

    def forward(self, x):
        x = self.patch_embed(x)
        return x


def _fuse_conv_bn(conv: nn.Module, bn: nn.Module) -> nn.Module:
    """Fuse conv and bn into one module.

    Args:
        conv (nn.Module): Conv to be fused.
        bn (nn.Module): BN to be fused.

    Returns:
        nn.Module: Fused module.
    """
    conv_w = conv.weight
    conv_b = conv.bias if conv.bias is not None else torch.zeros_like(
        bn.running_mean)

    factor = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    conv.weight = nn.Parameter(conv_w *
                               factor.reshape([conv.out_channels, 1, 1, 1]))
    conv.bias = nn.Parameter((conv_b - bn.running_mean) * factor + bn.bias)
    return conv


def fuse_conv_bn(module: nn.Module) -> nn.Module:
    """Recursively fuse conv and bn in a module.

    During inference, the functionary of batch norm layers is turned off
    but only the mean and var alone channels are used, which exposes the
    chance to fuse it with the preceding conv layers to save computations and
    simplify network structures.

    Args:
        module (nn.Module): Module to be fused.

    Returns:
        nn.Module: Fused module.
    """
    last_conv = None
    last_conv_name = None

    for name, child in module.named_children():
        if isinstance(child,
                      (nn.modules.batchnorm._BatchNorm, nn.SyncBatchNorm)):
            if last_conv is None:  # only fuse BN that is after Conv
                continue
            fused_conv = _fuse_conv_bn(last_conv, child)
            module._modules[last_conv_name] = fused_conv
            # To reduce changes, set BN as Identity instead of deleting it.
            module._modules[name] = nn.Identity()
            last_conv = None
        elif isinstance(child, nn.Conv2d):
            last_conv = child
            last_conv_name = name
        else:
            fuse_conv_bn(child)
    return module


class ConvolutionBatchNorm(Module):

    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size=3,
            stride=2,
            pad=1,
            dilation=1,
            groups=1,
    ):
        super(ConvolutionBatchNorm, self).__init__()
        self.conv = nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            dilation=dilation,
            groups=groups,
            bias=False)
        self.bn = BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

    @torch.no_grad()
    def fuse(self):
        return fuse_conv_bn(self).conv


class LinearBatchNorm(Module):

    def __init__(self, in_feature, out_feature,):
        super(LinearBatchNorm, self).__init__()
        self.linear = nn.Linear(in_feature, out_feature, bias=False)
        self.bn = BatchNorm1d(out_feature)

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x.flatten(0, 1)).reshape_as(x)
        return x

    @torch.no_grad()
    def fuse(self):
        w = self.bn.weight / (self.bn.running_var + self.bn.eps)**0.5
        w = self.linear.weight * w[:, None]
        b = self.bn.bias - self.bn.running_mean * self.bn.weight / \
            (self.bn.running_var + self.bn.eps) ** 0.5

        factory_kwargs = {
            'device': self.linear.weight.device,
            'dtype': self.linear.weight.dtype
        }
        bias = nn.Parameter(
            torch.empty(self.linear.out_features, **factory_kwargs))
        self.linear.register_parameter('bias', bias)
        self.linear.weight.data.copy_(w)
        self.linear.bias.data.copy_(b)
        return self.linear


class Residual(Module):

    def __init__(self, block, drop_path_rate=0.):
        super(Residual, self).__init__()
        self.block = block
        if drop_path_rate > 0:
            self.drop_path = DropPath(drop_path_rate)
        else:
            self.drop_path = nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.block(x))
        return x


class Attention(Module):

    def __init__(
            self,
            dim,
            key_dim,
            num_heads=8,
            attn_ratio=4,
            resolution=14,
    ):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2
        self.qkv = LinearBatchNorm(dim, h)
        self.proj = nn.Sequential(
            Hardswish(), LinearBatchNorm(self.dh, dim))

        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        """change the mode of model."""
        super(Attention, self).train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x (B,N,C)
        B, N, C = x.shape  # 2 196 128
        qkv = self.qkv(x)  # 2 196 128
        q, k, v = qkv.view(B, N, self.num_heads, -1).split(
            [self.key_dim, self.key_dim, self.d],
            dim=3)  # q 2 196 4 16 ; k 2 196 4 16; v 2 196 4 32
        q = q.permute(0, 2, 1, 3)  # 2 4 196 16
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = ((q @ k.transpose(-2, -1)) *
                self.scale  # 2 4 196 16 * 2 4 16 196 -> 2 4 196 196
                + (self.attention_biases[:, self.attention_bias_idxs]
                   if self.training else self.ab))
        attn = attn.softmax(dim=-1)  # 2 4 196 196 -> 2 4 196 196
        x = (attn @ v).transpose(1, 2).reshape(
            B, N,
            self.dh)  # 2 4 196 196 * 2 4 196 32 -> 2 4 196 32 -> 2 196 128
        x = self.proj(x)
        return x


class MLP(nn.Sequential):

    def __init__(self, embed_dim, mlp_ratio, ):
        super(MLP, self).__init__()
        h = embed_dim * mlp_ratio
        self.linear1 = LinearBatchNorm(embed_dim, h)
        self.activation = Hardswish()
        self.linear2 = LinearBatchNorm(h, embed_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class Subsample(Module):

    def __init__(self, stride, resolution):
        super(Subsample, self).__init__()
        self.stride = stride
        self.resolution = resolution

    def forward(self, x):
        B, _, C = x.shape
        # B, N, C -> B, H, W, C
        x = x.view(B, self.resolution, self.resolution, C)
        x = x[:, ::self.stride, ::self.stride]
        x = x.reshape(B, -1, C)  # B, H', W', C -> B, N', C
        return x


class AttentionSubsample(nn.Sequential):

    def __init__(self,
                 in_dim,
                 out_dim,
                 key_dim,
                 num_heads=8,
                 attn_ratio=2,
                 stride=2,
                 resolution=14):
        super(AttentionSubsample, self).__init__()
        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * self.num_heads
        self.attn_ratio = attn_ratio
        self.sub_resolution = (resolution - 1) // stride + 1
        h = self.dh + nh_kd
        self.kv = LinearBatchNorm(in_dim, h)

        self.q = nn.Sequential(
            Subsample(stride, resolution), LinearBatchNorm(in_dim, nh_kd))
        self.proj = nn.Sequential(
            Hardswish(), LinearBatchNorm(self.dh, out_dim))

        self.stride = stride
        self.resolution = resolution
        points = list(itertools.product(range(resolution), range(resolution)))
        sub_points = list(
            itertools.product(
                range(self.sub_resolution), range(self.sub_resolution)))
        N = len(points)
        N_sub = len(sub_points)
        attention_offsets = {}
        idxs = []
        for p1 in sub_points:
            for p2 in points:
                size = 1
                offset = (abs(p1[0] * stride - p2[0] + (size - 1) / 2),
                          abs(p1[1] * stride - p2[1] + (size - 1) / 2))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N_sub, N))

    @torch.no_grad()
    def train(self, mode=True):
        super(AttentionSubsample, self).train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):
        B, N, C = x.shape
        k, v = self.kv(x).view(B, N, self.num_heads,
                               -1).split([self.key_dim, self.d], dim=3)
        k = k.permute(0, 2, 1, 3)  # BHNC
        v = v.permute(0, 2, 1, 3)  # BHNC
        q = self.q(x).view(B, self.sub_resolution**2, self.num_heads,
                           self.key_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale + \
               (self.attention_biases[:, self.attention_bias_idxs]
                if self.training else self.ab)
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, -1, self.dh)
        x = self.proj(x)
        return x


class LeViT(Module):
    """LeViT backbone.

    A PyTorch implementation of `LeViT: A Vision Transformer in ConvNet's
    Clothing for Faster Inference <https://arxiv.org/abs/2104.01136>`_

    Modified from the official implementation:
    https://github.com/facebookresearch/LeViT

    Args:
        arch (str | dict): LeViT architecture.

            If use string, choose from '128s', '128', '192', '256' and '384'.
            If use dict, it should have below keys:

            - **embed_dims** (List[int]): The embed dimensions of each stage.
            - **key_dims** (List[int]): The embed dimensions of the key in the
              attention layers of each stage.
            - **num_heads** (List[int]): The number of heads in each stage.
            - **depths** (List[int]): The number of blocks in each stage.

        img_size (int): Input image size
        patch_size (int | tuple): The patch size. Deault to 16
        attn_ratio (int): Ratio of hidden dimensions of the value in attention
            layers. Defaults to 2.
        mlp_ratio (int): Ratio of hidden dimensions in MLP layers.
            Defaults to 2.
        act_cfg (dict): The config of activation functions.
            Defaults to ``dict(type='HSwish')``.
        hybrid_backbone (callable): A callable object to build the patch embed
            module. Defaults to use :class:`HybridBackbone`.
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        deploy (bool): Whether to switch the model structure to
            deployment mode. Defaults to False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """
    arch_zoo = {
        '128s': {
            'embed_dims': [128, 256, 384],
            'num_heads': [4, 6, 8],
            'depths': [2, 3, 4],
            'key_dims': [16, 16, 16],
        },
        '128': {
            'embed_dims': [128, 256, 384],
            'num_heads': [4, 8, 12],
            'depths': [4, 4, 4],
            'key_dims': [16, 16, 16],
        },
        '192': {
            'embed_dims': [192, 288, 384],
            'num_heads': [3, 5, 6],
            'depths': [4, 4, 4],
            'key_dims': [32, 32, 32],
        },
        '256': {
            'embed_dims': [256, 384, 512],
            'num_heads': [4, 6, 8],
            'depths': [4, 4, 4],
            'key_dims': [32, 32, 32],
        },
        '320': {
            'embed_dims': [320, 448, 640],
            'num_heads': [5, 7, 9],
            'depths': [4, 4, 4],
            'key_dims': [32, 32, 32],
        },
        '384': {
            'embed_dims': [384, 512, 768],
            'num_heads': [6, 9, 12],
            'depths': [4, 4, 4],
            'key_dims': [32, 32, 32],
        },
    }

    def __init__(self,
                 arch,
                 img_size=112,
                 patch_size=8,
                 attn_ratio=2,
                 mlp_ratio=2,
                 hybrid_backbone=HybridBackbone,
                 out_indices=-1,
                 deploy=False,
                 drop_path_rate=0,
                 num_features=512,
                 fp16=True):
        super(LeViT, self).__init__()

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch = self.arch_zoo[arch]
        elif isinstance(arch, dict):
            essential_keys = {'embed_dim', 'num_heads', 'depth', 'key_dim'}
            assert isinstance(arch, dict) and set(arch) == essential_keys, \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch = arch
        else:
            raise TypeError('Expect "arch" to be either a string '
                            f'or a dict, got {type(arch)}')

        self.embed_dims = self.arch['embed_dims']
        self.num_heads = self.arch['num_heads']
        self.key_dims = self.arch['key_dims']
        self.depths = self.arch['depths']
        self.num_stages = len(self.embed_dims)
        self.drop_path_rate = drop_path_rate

        self.fp16 = fp16
        self.num_features = num_features

        self.patch_embed = hybrid_backbone(self.embed_dims[0])

        self.resolutions = []
        resolution = img_size // patch_size
        self.stages = ModuleList()
        for i, (embed_dims, key_dims, depth, num_heads) in enumerate(
                zip(self.embed_dims, self.key_dims, self.depths,
                    self.num_heads)):
            blocks = []
            if i > 0:
                downsample = AttentionSubsample(
                    in_dim=self.embed_dims[i - 1],
                    out_dim=embed_dims,
                    key_dim=key_dims,
                    num_heads=self.embed_dims[i - 1] // key_dims,
                    attn_ratio=4,
                    stride=2,
                    resolution=resolution)
                blocks.append(downsample)
                resolution = downsample.sub_resolution
                if mlp_ratio > 0:  # mlp_ratio
                    blocks.append(
                        Residual(
                            MLP(embed_dims, mlp_ratio),
                            self.drop_path_rate))
            self.resolutions.append(resolution)
            for _ in range(depth):
                blocks.append(
                    Residual(
                        Attention(
                            embed_dims,
                            key_dims,
                            num_heads,
                            attn_ratio=attn_ratio,
                            resolution=resolution,
                        ), self.drop_path_rate))
                if mlp_ratio > 0:
                    blocks.append(
                        Residual(
                            MLP(embed_dims, mlp_ratio),
                            self.drop_path_rate))

            self.stages.append(Sequential(*blocks))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.linear = Linear(self.embed_dims[-1], self.num_features, bias=False)
        self.bn = BatchNorm1d(self.num_features)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        elif isinstance(out_indices, tuple):
            out_indices = list(out_indices)
        elif not isinstance(out_indices, list):
            raise TypeError('"out_indices" must by a list, tuple or int, '
                            f'get {type(out_indices)} instead.')
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_stages + index
            assert 0 <= out_indices[i] < self.num_stages, \
                f'Invalid out_indices {index}.'
        self.out_indices = out_indices

        self.apply(self._init_weights)

        self.deploy = False
        if deploy:
            self.switch_to_deploy()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def switch_to_deploy(self):
        if self.deploy:
            return
        fuse_parameters(self)
        self.deploy = True

    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)  # B, C, H, W -> B, L, C
        x = x.float() if self.fp16 else x
        for i, stage in enumerate(self.stages):
            x = stage(x)
        x = self.gap(x.permute((0, 2, 1))).view(x.size(0), -1)
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
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.zero_()


def fuse_parameters(module):
    for child_name, child in module.named_children():
        if hasattr(child, 'fuse'):
            setattr(module, child_name, child.fuse())
        else:
            fuse_parameters(child)


def get_levit_256(fp16, num_features):
    return LeViT(arch="256", fp16=fp16, num_features=num_features)

def get_levit_320(fp16, num_features):
    return LeViT(arch="320", fp16=fp16, num_features=num_features)
