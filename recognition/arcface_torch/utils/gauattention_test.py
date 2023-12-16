#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2022/2/25 14:21
# @Author  : SinGaln

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN


class GateAttentionUnit(nn.Module):
    def __init__(self, max_seq_length, hidden_size, expansion_factor=2, s=128, norm_type="layer_norm", eps=1e-5,
                 hidden_act="silu"):
        super(GateAttentionUnit, self).__init__()
        self.s = s
        self.max_seq_length = max_seq_length
        self.gamma = nn.Parameter(torch.rand((2, self.s)))
        self.beta = nn.Parameter(torch.rand((2, self.s)))
        self.e = int(hidden_size * expansion_factor)
        self.w = nn.Parameter(torch.rand([2 * max_seq_length - 1], dtype=torch.float))
        self.a = nn.Parameter(torch.rand([1, self.s], dtype=torch.float))
        self.b = nn.Parameter(torch.rand([1, self.s], dtype=torch.float))
        self.o = nn.Linear(self.e, hidden_size)
        self.uv = nn.Linear(hidden_size, 2 * self.e + self.s)
        self.LayerNorm = (
            nn.LayerNorm(hidden_size, eps=eps)
            if norm_type == "layer_norm"
            else self.ScaleNorm(eps=eps)
        )
        nn.init.xavier_uniform_(self.uv.weight)
        self.act_fn = ACT2FN[hidden_act]

    class ScaleNorm(nn.Module):
        def __init__(self, eps=1e-5):
            super().__init__()
            self.eps = eps
            self.scala = nn.Parameter(torch.ones(1))

        def forward(self, x):
            mean_square = (x ** 2).mean(dim=-1, keepdim=True)
            x = x * torch.rsqrt(mean_square + self.eps) * self.scala
            return x

    def rope(self, x, dim):
        """
        :param x: input tensor
        :param dim: opration dimension
        :return:
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

    def rel_pos_bias(self, seq_len):
        if seq_len <= 512:
            t = F.pad(self.w[: 2 * seq_len - 1], [0, seq_len]).repeat(seq_len)
            t = t[..., :-seq_len].reshape(-1, seq_len, 3 * seq_len - 2)
            r = (2 * seq_len - 1) // 2
            t = t[..., r:-r]
        else:
            # raise Exception("sequence length error.")
            a = self.rope(self.a.repeat(seq_len, 1), dim=0)
            b = self.rope(self.b.repeat(seq_len, 1), dim=0)
            t = torch.einsum("mk,nk->mn", a, b)
        return t

    def forward(self, x, attention_mask, causal=False):
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
        if attention_mask is not None:
             attention_mask = (1.0 - attention_mask) * -1e12
             qk = qk + attention_mask.unsqueeze(1)
        bias = self.rel_pos_bias(self.max_seq_length)[:, :seq_length, :seq_length]
        kernel = torch.square(F.relu(qk / self.max_seq_length + bias))

        if causal:
            causal_mask = torch.diagonal(torch.ones([seq_length, seq_length], dtype=torch.float))
            kernel *= causal_mask
        x = u * torch.einsum("bnm, bme->bne", kernel, v)
        x = self.o(x)
        return x + shortcut

if __name__=="__main__":
    x = torch.rand(32, 512, 768)
    attention_mask = torch.ones(32, 512)
    gau = GateAttentionUnit(512, 768)
    score = gau(x, attention_mask)

