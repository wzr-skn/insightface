"""
The main GhostNet architecture as specified in "GhostNet: More Features from Cheap Operations"
Paper:
https://arxiv.org/pdf/1911.11907.pdf
"""
from typing import Optional, Sequence

import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, Sequential, Module, ModuleList, Hardsigmoid
import torch
import torch.nn.functional as F

from torch.nn.modules.batchnorm import _BatchNorm

from .mobileone import MobileOneBlock, MobileOneBlock_pairs
from .mobilefacenet import ConvBlock, GDC


def _make_divisible(v, divisor=4, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class se_module(Module):
    def __init__(self, in_channel, se_ratio):
        super(se_module, self).__init__()
        self.in_channel = in_channel
        self.se_ratio = se_ratio
        self.squeeze_channels = _make_divisible(self.in_channel * self.se_ratio)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = Conv2d(
            in_channels=self.in_channel,
            out_channels=self.squeeze_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        self.act1 = PReLU(self.squeeze_channels)
        self.conv2 = Conv2d(
            in_channels=self.squeeze_channels,
            out_channels=self.in_channel,
            kernel_size=1,
            stride=1,
            bias=True)
        self.act2 = Hardsigmoid()

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.act1(self.conv1(out))
        out = self.act2(self.conv2(out))
        return x * out


class ghost_module(Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 convkernel=1,
                 dw_kernel=3,
                 add_activation=True):
        super(ghost_module, self).__init__()
        self.dw_kernel = dw_kernel
        self.add_activation = add_activation
        conv_out_channel = out_channel // 2
        self.conv1_1 = MobileOneBlock(
            in_channels=in_channel,
            out_channels=conv_out_channel,
            kernel_size=convkernel,
            num_convs=4,
            stride=1,
            padding=0,
            act=False)
        self.act1 = PReLU(conv_out_channel) if self.add_activation else None
        if self.dw_kernel > 3:
            self.dw_conv = Conv2d(
                in_channels=conv_out_channel,
                out_channels=conv_out_channel,
                kernel_size=dw_kernel,
                groups=conv_out_channel,
                stride=1,
                padding=dw_kernel//2,
                bias=False)
            self.norm2 = BatchNorm2d(conv_out_channel)
        else:
            self.dw_conv = MobileOneBlock(
                in_channels=conv_out_channel,
                out_channels=conv_out_channel,
                kernel_size=3,
                num_convs=4,
                stride=1,
                padding=1,
                groups=conv_out_channel,
                act=False)
        self.act2 = PReLU(conv_out_channel) if self.add_activation else None

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.act1(x) if self.add_activation else x
        gh_x = self.norm2(self.dw_conv(x)) if self.dw_kernel > 3 else self.dw_conv(x)
        gh_x = self.act2(gh_x) if self.add_activation else gh_x
        return torch.cat([x, gh_x], 1)


class ghost_bottleneck(Module):
    def __init__(self,
                 dw_kernel,
                 strides,
                 in_channel,
                 exp_channel,
                 out_channel,
                 se_ratio=0,
                 shortcut=True):
        super(ghost_bottleneck, self).__init__()
        self.in_channel = in_channel
        self.exp_channel = exp_channel
        self.out_channel = out_channel
        self.strides = strides
        self.se_ratio = se_ratio
        self.shortcut = shortcut

        self.ghost_module1 = ghost_module(self.in_channel, self.exp_channel, add_activation=True)
        if self.strides > 1:
            self.dw_conv1 = Conv2d(
                in_channels=self.exp_channel,
                out_channels=self.exp_channel,
                kernel_size=dw_kernel,
                groups=self.exp_channel,
                stride=self.strides,
                padding=dw_kernel//2,
                bias=False)
            self.norm1 = BatchNorm2d(self.exp_channel)

        if self.se_ratio > 0:
            self.se_module = se_module(self.exp_channel, self.se_ratio)

        self.ghost_module2 = ghost_module(self.exp_channel, self.out_channel, add_activation=False)

        if self.shortcut:
            self.dw_conv2 = Conv2d(
                in_channels=self.in_channel,
                out_channels=self.in_channel,
                kernel_size=dw_kernel,
                groups=self.in_channel,
                stride=self.strides,
                padding=dw_kernel//2,
                bias=False)
            self.norm2 = BatchNorm2d(self.in_channel)
            self.conv1_1 = MobileOneBlock(
                in_channels=self.in_channel,
                out_channels=self.out_channel,
                kernel_size=1,
                num_convs=4,
                stride=1,
                padding=0,
                act=False)

    def forward(self, x):
        out1 = self.ghost_module1(x)
        out1 = self.norm1(self.dw_conv1(out1)) if self.strides > 1 else out1
        out1 = self.se_module(out1) if self.se_ratio > 0 else out1
        out1 = self.ghost_module2(out1)
        if self.shortcut:
            out2 = self.dw_conv2(x)
            out2 = self.norm2(out2)
            out2 = self.conv1_1(out2)
        else:
            out2 = x
        return torch.add(out2, out1)


class GhostNet(Module):
    """GhostNet_v1 backbone"""
    arch_zoo = {
        's0':
            dict(
                dwkernels=[3, 3, 3, 5, 5, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5],
                exps=[16, 48, 72, 72, 120, 240, 200, 184, 184, 480, 672, 672, 960, 960, 960, 512],
                outs=[16, 24, 24, 40, 40, 80, 80, 80, 80, 112, 112, 160, 160, 160, 160, 160],
                use_ses=[0, 0, 0, 0.25, 0.25, 0, 0, 0, 0, 0.25, 0.25, 0.25, 0, 0.25, 0, 0.25],
                strides=[1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1]),
        's1':
            dict(
                dwkernels=[3, 3, 3, 5, 5, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5],
                exps=[24, 48, 72, 72, 120, 240, 200, 184, 184, 480, 672, 672, 960, 960, 960, 512],
                outs=[24, 32, 32, 60, 60, 100, 100, 100, 100, 160, 160, 200, 200, 200, 200, 200],
                use_ses=[0, 0, 0, 0.25, 0.25, 0, 0, 0, 0, 0.25, 0.25, 0.25, 0, 0.25, 0, 0.25],
                strides=[1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1]),
    }

    def __init__(self,
                 arch="s1",
                 in_channels=3,
                 frozen_stages=-1,
                 num_features=512,
                 norm_eval=False,
                 fp16=True,
                 width=1.3):
        super(GhostNet, self).__init__()

        if isinstance(arch, str):
            assert arch in self.arch_zoo, f'"arch": "{arch}"' \
                f' is not one of the {list(self.arch_zoo.keys())}'
            arch = self.arch_zoo[arch]
        elif not isinstance(arch, dict):
            raise TypeError('Expect "arch" to be either a string '
                            f'or a dict, got {type(arch)}')

        self.arch = arch

        self.in_channels = in_channels
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.fp16 = fp16
        self.num_features = num_features

        channels = min(64, _make_divisible(16 * width, 4))
        self.stage0 = MobileOneBlock(
            self.in_channels,
            channels,
            stride=2,
            kernel_size=3,
            num_convs=4)

        self.in_planes = channels
        self.stages = []
        for i, dw_kernel in enumerate(self.arch['dwkernels']):
            out = _make_divisible(self.arch['outs'][i] * width, 4)  # [ 20 32 32 52 52 104 104 104 104 144 144 208 208 208 208 208 ]
            exp = _make_divisible(self.arch['exps'][i] * width, 4)
            se = self.arch['use_ses'][i]
            stride = self.arch['strides'][i]

            shortcut = False if out == self.in_planes and stride == 1 else True

            stage = ghost_bottleneck(dw_kernel, stride, self.in_planes, exp, out, se, shortcut)

            stage_name = f'stage{i + 1}'
            self.add_module(stage_name, stage)
            self.stages.append(stage_name)
            self.in_planes = out

        self.conv_sep = ConvBlock(self.in_planes, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.features = GDC(self.num_features)
        self._initialize_weights()

    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            x = self.stage0(x)
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
        super(GhostNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

def get_gh_s0(fp16, num_features):
    return GhostNet(arch="s0", fp16=fp16, num_features=num_features,)

def get_gh_s1(fp16, num_features):
    return GhostNet(arch="s1", fp16=fp16, num_features=num_features,)

