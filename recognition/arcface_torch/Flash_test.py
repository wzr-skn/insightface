#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2022/3/16 11:59
# @Author  : SinGaln
"""
    FLASH: https://arxiv.org/abs/2202.10447
    融合了Attention和FFN的新Transformer的变体，速度快于传统的Transformer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN


# 对输入进行缩放
class ScaleNorm(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scala = nn.Parameter(torch.ones(1))

    def forward(self, x):
        mean_square = (x ** 2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(mean_square + self.eps) * self.scala
        return x


# RoPE位置编码
def rope(x, dim):
    """
    :param x: input tensor
    :param dim: oprate dimension
    :return: tensor
    """
    shape = x.shape
    if isinstance(dim, int):
        dim = [dim]

    spatial_shape = [shape[i] for i in dim]
    total_len = 1
    for i in spatial_shape:
        total_len *= i
    position = torch.reshape(torch.arange(total_len, dtype=torch.float, device=x.device), spatial_shape)

    for i in range(dim[-1] + 1, len(shape) - 1, 1):
        position = torch.unsqueeze(position, dim=-1)

    half_size = shape[-1] // 2
    freq_seq = -torch.arange(half_size, dtype=torch.float, device=x.device) / float(half_size)
    inv_freq = 10000 ** -freq_seq
    sinusoid = torch.einsum("...,d->...d", position, inv_freq)
    sin = torch.sin(sinusoid)
    cos = torch.cos(sinusoid)
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


# 相对位置编码
def rel_pos_bias(seq_len, s):
    a = torch.rand([1, s], dtype=torch.float)
    b = torch.rand([1, s], dtype=torch.float)
    w = torch.rand([2 * seq_len - 1], dtype=torch.float)
    if seq_len <= 512:
        t = F.pad(w[: 2 * seq_len - 1], [0, seq_len]).repeat(seq_len)
        t = t[..., :-seq_len].reshape(-1, seq_len, 3 * seq_len - 2)
        r = (2 * seq_len - 1) // 2
        t = t[..., r:-r]
    else:
        a = rope(a.repeat(seq_len, 1), dim=[0])
        b = rope(b.repeat(seq_len, 1), dim=[0])
        t = torch.einsum("mk,nk->mn", a, b)
    return t


# GAU单元
class GateAttentionUnit(nn.Module):
    """
    GAU Block: Gate Attention Unit
    """

    def __init__(self, max_seq_length, hidden_size, expansion_factor=2, s=128, norm_type="layer_norm", eps=1e-5,
                 hidden_act="silu"):
        super(GateAttentionUnit, self).__init__()
        self.s = s
        self.max_seq_length = max_seq_length
        self.gamma = nn.Parameter(torch.rand((2, self.s)))
        self.beta = nn.Parameter(torch.rand((2, self.s)))
        self.e = int(hidden_size * expansion_factor)
        self.o = nn.Linear(self.e, hidden_size)
        self.uv = nn.Linear(hidden_size, 2 * self.e + self.s)
        self.LayerNorm = (nn.LayerNorm(hidden_size, eps=eps) if norm_type == "layer_norm" else ScaleNorm(eps=eps))

        nn.init.xavier_uniform_(self.uv.weight)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x, causal=False):
        """
        :param x:  [batch_size, sequence_length, model_dim]
        :param causal:add mask tensor matrix
        :return:
        """
        seq_length = x.shape[1]
        shortcut, x = x, self.LayerNorm(x)

        uv = self.uv(x)
        u, v, base = torch.split(self.act_fn(uv), [self.e, self.e, self.s], dim=-1)
        base = torch.einsum("...r, hr->...hr", base, self.gamma) + self.beta
        base = self.rope(base, dim=1)
        q, k = torch.unbind(base, dim=-2)

        qk = torch.einsum("bnd,bmd->bnm", q, k)
        bias = rel_pos_bias(self.max_seq_length, self.s)[:, :seq_length, :seq_length]
        kernel = torch.square(F.relu(qk / self.max_seq_length + bias))

        if causal:
            causal_mask = torch.diagonal(torch.ones([seq_length, seq_length], dtype=torch.float))
            kernel *= causal_mask
        x = u * torch.einsum("bnm, bme->bne", kernel, v)
        x = self.o(x)
        out = x + shortcut
        return out


# FLASH单元
class FlashModel(nn.Module):
    """
    FLASH Block: Fast Linear Attention with a Single Head
    """

    def __init__(self, model_size, sequence_length, expansion_factor=2, s=128, norm_type="layer_norm", eps=1e-5,
                 hidden_act="silu"):
        super(FlashModel, self).__init__()
        self.s = s
        self.eps = eps
        self.norm_type = norm_type
        self.model_size = model_size
        self.hidden_act = hidden_act
        self.sequence_length = sequence_length
        self.expansion_factor = expansion_factor
        self.e = int(self.model_size * self.expansion_factor)

        self.dense1 = nn.Linear(self.model_size, 2 * self.e + self.s, bias=True)
        self.gamma = nn.Parameter(torch.rand((4, self.s)))
        self.beta = nn.Parameter(torch.rand((4, self.s)))
        self.dense2 = nn.Linear(self.e, self.model_size)
        self.LayerNorm = (
            nn.LayerNorm(model_size, eps=self.eps) if norm_type == "layer_norm" else ScaleNorm(eps=self.eps))

        nn.init.xavier_normal_(self.dense1.weight)
        self.act_fn = ACT2FN[self.hidden_act]

    def global_linear_attention(self, query, key, value, causal):
        if causal:
            kv = torch.einsum("bgcs, bgce->bgse", key, value)
            kv = torch.cumsum(kv, dim=1)
            lin_v = torch.einsum("bgcs, bgse->bgce", query, kv)
            return lin_v
        else:
            kv = torch.einsum("bgcs, bgce->bse", key, value)
            lin_v = torch.einsum("bgcs, bse->bgce", query, kv)
            return lin_v

    def segment_ids_to_mask(self, segment_ids, causal=False):
        """Generate the segment mask from the segment ids.
        The segment mask is used to remove the attention between tokens in different documents.
        """
        min_ids, max_ids = torch.min(segment_ids, dim=-1).values, torch.max(segment_ids, dim=-1).values
        # 1.0 indicates in the same group and 0.0 otherwise
        mask = torch.logical_and(torch.less_equal(min_ids[:, :, None], max_ids[:, None, :]),
                                 torch.greater_equal(max_ids[:, :, None], min_ids[:, None, :]))
        mask = torch.tensor(mask, dtype=torch.float32)
        if causal:
            g = segment_ids.size()[1]
            causal_mask = 1.0 - torch.triu(torch.ones([g, g], dtype=torch.float32))  # 保留主对角线以及主对角线以上的元素
            mask *= causal_mask
        mask = torch.div(mask, torch.sum(mask, dim=-1, keepdim=True))
        return mask

    def forward(self, inputs, segment_ids, causal=False):
        """
        inputs: [batch_size, num_chunk, chunk_length, model_size]
        """
        _, g, n, d = inputs.size()
        shortcut, inputs = inputs, self.LayerNorm(inputs)
        # 通过线性变换得到Z，见论文公式(4)
        uv = self.dense1(inputs)
        # 将uv按最后一维切分，得到Ug:[C*e],Vg:[C*e], Zg:[C*s], 论文中的3.2部分
        # u:[batch_size, num_chunk, chunk_length, self.e]
        # v:[batch_size, num_chunk, chunk_length, self.e]
        # z:[batch_size, num_chunk, chunk_length, self.s]
        u, v, z = torch.split(self.act_fn(uv), [self.e, self.e, self.s], dim=-1)

        # 生成quad_q, quad_k, lin_q, lin_k
        # 首先进行简单的offset和scale,融入RoPE位置向量
        z = torch.einsum("...r, hr->...hr", z, self.gamma) + self.beta
        z = rope(z, dim=[1, 2])
        quad_q, quad_k, lin_q, lin_k = torch.unbind(z, dim=-2)  # 按-2维进行分解得到quad_q, quad_k, lin_q和lin_k
        # 计算global的lin_v
        lin_v = self.global_linear_attention(lin_q, lin_k, v, causal)
        if causal:
            # 线性注意力部分
            lin_kv = torch.einsum("bgnk, bgne->bgke", lin_k, lin_v) / torch.tensor(n, inputs.dtype)  # 见公式(7)
            mask = self.segment_ids_to_mask(segment_ids=segment_ids, causal=causal)
            cum_lin_kv = torch.einsum('bhke, bgh->bgke', lin_kv, mask)
            linear = torch.einsum("bgnk, bgke->bgne", lin_kv, cum_lin_kv)
            # 二次注意力
            quad_qk = torch.einsum("bgnk, bgmk->bgnm", quad_q, quad_k)  # 论文Local attention per chunk部分
            bias = rel_pos_bias(self.sequence_length, self.s)[:, :n, :n]
            kernel = torch.square(F.relu(quad_qk / n + bias))  # 论文中的relu**2部分
            causal_mask = torch.triu(torch.ones([n, n], dtype=inputs.dtype))
            quadratic = torch.einsum("bgnm, bgme->bgne", kernel * causal_mask, v)
        else:
            lin_kv = torch.einsum("bgnk, bgne->bgke", lin_k, lin_v) / torch.tensor(n, dtype=inputs.dtype)  # 见公式(7)
            mask = self.segment_ids_to_mask(segment_ids=segment_ids, causal=causal)
            lin_kv = torch.einsum("bhke, bgh->bgke", lin_kv, mask)
            linear = torch.einsum("bgnk, bgke->bgne", lin_q, lin_kv)
            # 二次注意力
            quad_qk = torch.einsum("bgnk, bgmk->bgnm", quad_q, quad_k)  # 论文Local attention per chunk部分
            bias = rel_pos_bias(self.sequence_length, self.s)[:, :n, :n]
            kernel = torch.square(F.relu(quad_qk / n + bias))  # 论文中的relu**2部分
            quadratic = torch.einsum("bgnm, bgme->bgne", kernel, v)
        inputs = u * (quadratic + linear)
        inputs = self.dense2(inputs)
        outputs = inputs + shortcut
        return outputs


if __name__=="__main__":
    x = torch.rand(32, 4, 64, 768)
    attention_mask = torch.ones(32, 4, 64)
    Flash = FlashModel(768, 256)
    score = Flash(x, attention_mask)

