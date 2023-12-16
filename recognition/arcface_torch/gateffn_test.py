import math
import torch
import torch.nn as nn
from transformers.activations import ACT2FN

class MultiheadAttention(nn.Module):
    def __init__(self, model_dim, num_head, dropout_rate):
        super(MultiheadAttention, self).__init__()
        self.model_dim = model_dim
        self.num_head = num_head
        if self.model_dim % self.num_head != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (self.model_dim, self.num_head))

        self.attention_head_size = int(self.model_dim / self.num_head)
        self.all_head_size = self.num_head * self.attention_head_size

        self.query = nn.Linear(self.model_dim, self.all_head_size)
        self.key = nn.Linear(self.model_dim, self.all_head_size)
        self.value = nn.Linear(self.model_dim, self.all_head_size)

        self.dropout = nn.Dropout(dropout_rate)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_head,
                                       self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # 计算q*k^T，并且按照原始attention进行scale
        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # 判断attention_mask是否存在
        if attention_mask is None:
            attention_mask = torch.ones_like(attention_scores)
        attention_scores = attention_scores + attention_mask

        # 进行归一化
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # dropout
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
                                  :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

class GateFFNDense(nn.Module):
    def __init__(self, model_dim, dropout_rate, hidden_unit=2048):
        super(GateFFNDense, self).__init__()
        """
        对应论文中的W，V，以及W2，激活函数对应GELU_new
        """
        self.W = nn.Linear(model_dim, hidden_unit, bias=False)
        self.V = nn.Linear(model_dim, hidden_unit, bias=False)
        self.W2 = nn.Linear(hidden_unit, model_dim, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.gelu_act = ACT2FN["gelu_new"]

    def forward(self, hidden_states):
        hidden_gelu = self.gelu_act(self.W(hidden_states))
        hidden_linear = self.V(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.W2(hidden_states)
        return hidden_states

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        对应TransformerBlock中的LayerNorm归一化
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # 对hidden_states平方后最后一维求均值，也就是求方差
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        # 求出标准差后乘以hidde_states
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # 将类型转为半精度
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states

class GateFFNLayer(nn.Module):
    def __init__(self, model_dim, dropout_rate):
        super(GateFFNLayer, self).__init__()
        """
        对应原始Transformer的FFN层
        """
        self.DenseReluDense = GateFFNDense(model_dim, dropout_rate)
        self.layer_norm = LayerNorm(model_dim, eps=1e-8)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states

class TransformerBlock(nn.Module):
    def __init__(self, model_dim, num_head, dropout_rate):
        super(TransformerBlock, self).__init__()
        self.model_dim = model_dim

        self.multiattention = MultiheadAttention(model_dim, num_head, dropout_rate)
        self.gateffn = GateFFNLayer(model_dim, dropout_rate)

    def forward(self, inputs):
        output = self.multiattention(inputs)
        output = self.gateffn(output)
        return output

if __name__=="__main__":
    x = torch.randn(32, 512, 768)
    tb = TransformerBlock(model_dim=768, num_head=12, dropout_rate=0.2)
    out = tb(x)