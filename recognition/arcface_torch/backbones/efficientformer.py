# Copyright (c) OpenMMLab. All rights reserved.
import itertools
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, LayerNorm, GroupNorm, PReLU, Sequential, Module, ModuleList

from .mobileone import MobileOneBlock, MobileOneBlock_pairs


def drop_path(x: torch.Tensor,
              drop_prob: float = 0.,
              training: bool = False) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample (when applied in main path of
    residual blocks).

    We follow the implementation
    https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py  # noqa: E501
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # handle tensors with different dimensions, not just 4D tensors.
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(
        shape, dtype=x.dtype, device=x.device)
    output = x.div(keep_prob) * random_tensor.floor()
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of
    residual blocks).

    We follow the implementation
    https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py  # noqa: E501

    Args:
        drop_prob (float): Probability of the path to be zeroed. Default: 0.1
    """

    def __init__(self, drop_prob: float = 0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class LayerScale(nn.Module):
    """LayerScale layer.

    Args:
        dim (int): Dimension of input features.
        layer_scale_init_value (float or torch.Tensor): Init value of layer
            scale. Defaults to 1e-5.
        inplace (bool): inplace: can optionally do the
            operation in-place. Defaults to False.
        data_format (str): The input data format, could be 'channels_last'
             or 'channels_first', representing (B, C, H, W) and
             (B, N, C) format data respectively. Defaults to 'channels_last'.
    """

    def __init__(self,
                 dim: int,
                 layer_scale_init_value: Union[float, torch.Tensor] = 1e-5,
                 inplace: bool = False,
                 data_format: str = 'channels_last'):
        super().__init__()
        assert data_format in ('channels_last', 'channels_first'), \
            "'data_format' could only be channels_last or channels_first."
        self.inplace = inplace
        self.data_format = data_format
        self.weight = nn.Parameter(torch.ones(dim) * layer_scale_init_value)

    def forward(self, x):
        if self.data_format == 'channels_first':
            if self.inplace:
                return x.mul_(self.weight.view(-1, 1, 1))
            else:
                return x * self.weight.view(-1, 1, 1)
        return x.mul_(self.weight) if self.inplace else x * self.weight


class AttentionWithBias(Module):
    """Multi-head Attention Module with attention_bias.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads. Defaults to 8.
        key_dim (int): The dimension of q, k. Defaults to 32.
        attn_ratio (float): The dimension of v equals to
            ``key_dim * attn_ratio``. Defaults to 4.
        resolution (int): The height and width of attention_bias.
            Defaults to 7.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads=8,
                 key_dim=32,
                 attn_ratio=4.,
                 resolution=7):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.attn_ratio = attn_ratio
        self.key_dim = key_dim
        self.nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        h = self.dh + self.nh_kd * 2
        self.qkv = nn.Linear(embed_dims, h)
        self.proj = nn.Linear(self.dh, embed_dims)

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
        self.attention_biases = nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        """change the mode of model."""
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):
        """forward function.

        Args:
            x (tensor): input features with shape of (B, N, C)
        """
        B, N, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.d], dim=-1)

        attn = ((q @ k.transpose(-2, -1)) * self.scale +
                (self.attention_biases[:, self.attention_bias_idxs]
                 if self.training else self.ab))
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dh)
        x = self.proj(x)
        return x


class Flat(Module):
    """Flat the input from (B, C, H, W) to (B, H*W, C)."""

    def __init__(self, ):
        super().__init__()

    def forward(self, x: torch.Tensor):
        x = x.flatten(2).transpose(1, 2)
        return x


class LinearMlp(Module):
    """Mlp implemented with linear.

    The shape of input and output tensor are (B, N, C).

    Args:
        in_features (int): Dimension of input features.
        hidden_features (int): Dimension of hidden features.
        out_features (int): Dimension of output features.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='BN')``.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        drop (float): Dropout rate. Defaults to 0.0.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 in_features: int,
                 hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): input tensor with shape (B, N, C).

        Returns:
            torch.Tensor: output tensor with shape (B, N, C).
        """
        x = self.drop1(self.act(self.fc1(x)))
        x = self.drop2(self.fc2(x))
        return x


class ConvMlp(Module):
    """Mlp implemented with 1*1 convolutions.

    Args:
        in_features (int): Dimension of input features.
        hidden_features (int): Dimension of hidden features.
        out_features (int): Dimension of output features.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='BN')``.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        drop (float): Dropout rate. Defaults to 0.0.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = MobileOneBlock(
                in_channels=in_features,
                out_channels=hidden_features,
                kernel_size=1,
                num_convs=1,
                stride=1,
                padding=0)
        self.fc2 = MobileOneBlock(
                in_channels=hidden_features,
                out_channels=out_features,
                kernel_size=1,
                num_convs=1,
                stride=1,
                padding=0,
                act=False)

        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): input tensor with shape (B, C, H, W).

        Returns:
            torch.Tensor: output tensor with shape (B, C, H, W).
        """

        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Meta3D(Module):
    """Meta Former block using 3 dimensions inputs, ``torch.Tensor`` with shape
    (B, N, C)."""

    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 use_layer_scale=True):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.token_mixer = AttentionWithBias(dim)
        self.norm2 = LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = LinearMlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        if use_layer_scale:
            self.ls1 = LayerScale(dim)
            self.ls2 = LayerScale(dim)
        else:
            self.ls1, self.ls2 = nn.Identity(), nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.ls1(self.token_mixer(self.norm1(x))))
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))
        return x


class Meta4D(Module):
    """Meta Former block using 4 dimensions inputs, ``torch.Tensor`` with shape
    (B, C, H, W)."""

    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 use_layer_scale=True,
                 fp16=True):
        super().__init__()

        self.token_mixer = MobileOneBlock_pairs(dim, dim, pw_act=False)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ConvMlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        if use_layer_scale:
            self.ls1 = LayerScale(dim, data_format='channels_first')
            self.ls2 = LayerScale(dim, data_format='channels_first')
        else:
            self.ls1, self.ls2 = nn.Identity(), nn.Identity()

        self.fp16 = fp16

    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            x = x + self.drop_path(self.ls1(self.token_mixer(x)))
            x = x + self.drop_path(self.ls2(self.mlp(x)))
        return x


def basic_blocks(in_channels,
                 out_channels,
                 index,
                 layers,
                 num_conv_branches=1,
                 mlp_ratio=4.,
                 drop_rate=.0,
                 drop_path_rate=0.,
                 use_layer_scale=True,
                 vit_num=1,
                 vit_form='Meta3D',
                 has_downsamper=False,
                 fp16=True):
    """generate EfficientFormer blocks for a stage."""
    blocks = []
    if has_downsamper:
        blocks.append(MobileOneBlock_pairs(in_channels, out_channels, stride=2))

    if vit_num == layers[index]:
        blocks.append(Flat())
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (
            sum(layers) - 1)
        if layers[index] - block_idx <= vit_num and vit_form == 'Meta3D':
            blocks.append(
                Meta3D(
                    out_channels,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=block_dpr,
                    use_layer_scale=use_layer_scale,
                ))
        elif layers[index] - block_idx <= vit_num and vit_form == 'GAU':
            from .flash import GAUBlock
            blocks.append(
                GAUBlock(
                    49,
                    out_channels,
                    out_channels,
                    expansion_factor=2,
                    use_rel_bias=False,
                    pos_enc=False
                ))
        else:
            blocks.append(
                Meta4D(
                    out_channels,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=block_dpr,
                    use_layer_scale=use_layer_scale,
                    fp16=fp16))
            if vit_num != 0 and layers[index] - block_idx - 1 == vit_num:
                blocks.append(Flat())
    blocks = nn.Sequential(*blocks)
    return blocks


class EfficientFormer(Module):
    """EfficientFormer.

    A PyTorch implementation of EfficientFormer introduced by:
    `EfficientFormer: Vision Transformers at MobileNet Speed <https://arxiv.org/abs/2206.01191>`_

    Modified from the `official repo
    <https://github.com/snap-research/EfficientFormer>`.

    Args:
        arch (str | dict): The model's architecture. If string, it should be
            one of architecture in ``EfficientFormer.arch_settings``. And if dict,
            it should include the following 4 keys:

            - layers (list[int]): Number of blocks at each stage.
            - embed_dims (list[int]): The number of channels at each stage.
            - downsamples (list[int]): Has downsample or not in the four stages.
            - vit_num (int): The num of vit blocks in the last stage.

            Defaults to 'l1'.

        in_channels (int): The num of input channels. Defaults to 3.
        pool_size (int): The pooling size of ``Meta4D`` blocks. Defaults to 3.
        mlp_ratios (int): The dimension ratio of multi-head attention mechanism
            in ``Meta4D`` blocks. Defaults to 3.
        reshape_last_feat (bool): Whether to reshape the feature map from
            (B, N, C) to (B, C, H, W) in the last stage, when the ``vit-num``
            in ``arch`` is not 0. Defaults to False. Usually set to True
            in downstream tasks.
        out_indices (Sequence[int]): Output from which stages.
            Defaults to -1.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Defaults to -1.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        drop_rate (float): Dropout rate. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        use_layer_scale (bool): Whether to use use_layer_scale in MetaFormer
            block. Defaults to True.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.

    Example:
        >>> import torch
        >>> inputs = torch.rand((1, 3, 224, 224))
        >>> # build EfficientFormer backbone for classification task
        >>> model = EfficientFormer(arch="l1")
        >>> model.eval()
        >>> level_outputs = model(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 448, 49)
        >>> # build EfficientFormer backbone for downstream task
        >>> model = EfficientFormer(
        >>>    arch="l3",
        >>>    out_indices=(0, 1, 2, 3),
        >>>    reshape_last_feat=True)
        >>> model.eval()
        >>> level_outputs = model(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 64, 56, 56)
        (1, 128, 28, 28)
        (1, 320, 14, 14)
        (1, 512, 7, 7)
    """  # noqa: E501

    # --layers: [x,x,x,x], numbers of layers for the four stages
    # --embed_dims: [x,x,x,x], embedding dims for the four stages
    # --downsamples: [x,x,x,x], has downsample or not in the four stages
    # --vit_num：(int), the num of vit blocks in the last stage
    arch_settings = {
        'l1': {
            'layers': [3, 2, 6, 4],
            'embed_dims': [48, 96, 224, 448],
            'downsamples': [False, True, True, True],
            'vit_num': [0, 0, 0, 1],
            'vit_form': 'Meta3D'
        },
        'l3': {
            'layers': [4, 4, 12, 6],
            'embed_dims': [64, 128, 320, 512],
            'downsamples': [False, True, True, True],
            'vit_num': [0, 0, 0, 4],
            'vit_form': 'Meta3D'
        },
        'l3_shallow': {
            'layers': [3, 3, 9, 4],
            'embed_dims': [64, 128, 256, 512],
            'downsamples': [False, True, True, True],
            'vit_num': [0, 0, 0, 4],
            'vit_form': 'Meta3D'
        },
        'l3_medium_GAU': {
            'layers': [3, 3, 9, 9],
            'embed_dims': [64, 128, 256, 512],
            'downsamples': [False, True, True, True],
            'vit_num': [0, 0, 0, 9],
            'vit_form': 'GAU'
        },
        'l3_GAU': {
            'layers': [3, 3, 9, 6],
            'embed_dims': [64, 128, 256, 512],
            'downsamples': [False, True, True, True],
            'vit_num': [0, 0, 0, 6],
            'vit_form': 'GAU'
        },
        'l3_shallow_GAU': {
            'layers': [3, 3, 9, 4],
            'embed_dims': [64, 128, 256, 512],
            'downsamples': [False, True, True, True],
            'vit_num': [0, 0, 0, 4],
            'vit_form': 'GAU'
        },
        'l3_shallow_GAU_2b': {
            'layers': [3, 3, 6, 4],
            'embed_dims': [64, 128, 256, 512],
            'downsamples': [False, True, True, True],
            'vit_num': [0, 0, 6, 4],
            'vit_form': 'GAU'
        },
        'l7': {
            'layers': [6, 6, 18, 8],
            'embed_dims': [96, 192, 384, 768],
            'downsamples': [False, True, True, True],
            'vit_num': [0, 0, 0, 8],
            'vit_form': 'Meta3D'
        },
    }

    def __init__(self,
                 arch='l1',
                 in_channels=3,
                 mlp_ratios=4,
                 reshape_last_feat=False,
                 out_indices=-1,
                 frozen_stages=-1,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 use_layer_scale=True,
                 num_features=512,
                 fp16=True):

        super().__init__()
        self.num_extra_tokens = 0  # no cls_token, no dist_token

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            default_keys = set(self.arch_settings['l1'].keys())
            assert set(arch.keys()) == default_keys, \
                f'The arch dict must have {default_keys}, ' \
                f'but got {list(arch.keys())}.'

        self.layers = arch['layers']
        self.embed_dims = arch['embed_dims']
        self.downsamples = arch['downsamples']
        assert isinstance(self.layers, list) and isinstance(
            self.embed_dims, list) and isinstance(self.downsamples, list)
        assert len(self.layers) == len(self.embed_dims) == len(
            self.downsamples)

        self.vit_num = arch['vit_num']
        self.vit_form = arch['vit_form']
        self.reshape_last_feat = reshape_last_feat

        self.fp16 = fp16
        self.num_features = num_features

        # assert self.vit_num >= 0, "'vit_num' must be an integer " \
        #                           'greater than or equal to 0.'
        # assert self.vit_num <= self.layers[-1], (
        #     "'vit_num' must be an integer smaller than layer number")

        self._make_stem(in_channels, self.embed_dims[0])

        # set the main block in network
        network = []
        for i in range(len(self.layers)):
            if i != 0:
                in_channels = self.embed_dims[i - 1]
            else:
                in_channels = self.embed_dims[i]
            out_channels = self.embed_dims[i]
            stage = basic_blocks(
                in_channels,
                out_channels,
                i,
                self.layers,
                mlp_ratio=mlp_ratios,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                vit_num=self.vit_num[i],
                vit_form=self.vit_form,
                use_layer_scale=use_layer_scale,
                has_downsamper=self.downsamples[i],
                fp16=self.fp16)
            network.append(stage)

        self.network = ModuleList(network)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'

        self.out_indices = out_indices
        for i_layer in self.out_indices:
            if not self.reshape_last_feat and \
                    i_layer == 3 and self.vit_num[i_layer] > 0:
                layer = LayerNorm(self.embed_dims[i_layer])
            else:
                # use GN with 1 group as channel-first LN2D
                layer = GroupNorm(num_groups=1, num_channels=self.embed_dims[i_layer])

            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.linear = Linear(self.embed_dims[-1], self.num_features, bias=False)
        self.bn = BatchNorm1d(self.num_features)

        self.frozen_stages = frozen_stages
        self._freeze_stages()
        self.apply(self._init_weights)

    def _make_stem(self, in_channels: int, stem_channels: int):
        """make 2-ConvBNReLu stem layer."""
        self.patch_embed = MobileOneBlock(
            in_channels,
            stem_channels,
            stride=2,
            kernel_size=3,
            num_convs=1,
            deploy=False)

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            N, _, H, W = x.shape
            if self.downsamples[idx]:
                H, W = H // 2, W // 2
            x = block(x)
            if self.vit_num[idx] > 0 and idx != len(self.vit_num) - 1:
                x = x.permute((0, 2, 1)).reshape(N, -1, H, W)
            if idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')

                if idx == len(self.network) - 1 and x.dim() == 3:
                    # when ``vit-num`` > 0 and in the last stage,
                    # if `self.reshape_last_feat`` is True, reshape the
                    # features to `BCHW` format before the final normalization.
                    # if `self.reshape_last_feat`` is False, do
                    # normalization directly and permute the features to `BCN`.
                    if self.reshape_last_feat:
                        x = x.permute((0, 2, 1)).reshape(N, -1, H, W)
                        x = norm_layer(x)
                    else:
                        x = norm_layer(x).permute((0, 2, 1))
                else:
                    x = norm_layer(x)

                x = self.gap(x).view(x.size(0), -1)
                x_out = self.bn(self.linear(x))

        return x_out.contiguous()

    def forward(self, x):
        # input embedding
        x = self.patch_embed(x)
        # through stages
        x = self.forward_tokens(x)
        return x

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
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
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
        super(EfficientFormer, self).train(mode)
        self._freeze_stages()

def get_ef_l7(fp16, num_features):
    return EfficientFormer(arch="l7", fp16=fp16, num_features=num_features)

def get_ef_l3(fp16, num_features):
    return EfficientFormer(arch="l3", fp16=fp16, num_features=num_features)

def get_ef_l3_shallow(fp16, num_features):
    return EfficientFormer(arch="l3_shallow", fp16=fp16, num_features=num_features, mlp_ratios=2,)

def get_ef_l3_shallow_GAU(fp16, num_features):
    return EfficientFormer(arch="l3_shallow_GAU", fp16=fp16, num_features=num_features, mlp_ratios=2,)

def get_ef_l3_GAU(fp16, num_features):
    return EfficientFormer(arch="l3_GAU", fp16=fp16, num_features=num_features, mlp_ratios=2,)

def get_ef_l3_shallow_GAU_2b(fp16, num_features):
    return EfficientFormer(arch="l3_shallow_GAU_2b", fp16=fp16, num_features=num_features, mlp_ratios=2,)

def get_ef_l3_medium_GAU(fp16, num_features):
    return EfficientFormer(arch="l3_medium_GAU", fp16=fp16, num_features=num_features, mlp_ratios=2,)
