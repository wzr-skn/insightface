import torch
import torch.nn as nn

class GateLinearUnit(nn.Module):
    def __init__(self, embedding_size, num_filers, kernel_size, vocab_size, bias=True, batch_norm=True, activation=nn.Tanh()):
        super(GateLinearUnit, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.conv_layer1 = nn.Conv2d(1, num_filers, (kernel_size, embedding_size), bias=bias)
        self.conv_layer2 = nn.Conv2d(1, num_filers, (kernel_size, embedding_size), bias=bias)
        self.batch_norm = nn.BatchNorm2d(num_filers)
        self.sigmoid = nn.Sigmoid()

        nn.init.kaiming_uniform_(self.conv_layer1.weight)
        nn.init.kaiming_uniform_(self.conv_layer2.weight)

    def gate(self, inputs):
        """门控机制"""
        return self.sigmoid(inputs)

    def forward(self, inputs):
        embed = self.embedding(inputs)
        embed = embed.unsqueeze(1)
        output = self.conv_layer1(embed)
        gate_output = self.conv_layer2(embed)
        # Gate Operation
        if self.activation is not None:
            # GTU
            output = self.activation(output) * self.gate(gate_output)
        else:
            # GLU
            output = output * self.gate(gate_output)
        if self.batch_norm:
            output = self.batch_norm(output)
            output = output.squeeze()
            return output
        else:
            return output.squeeze()

if __name__=="__main__":
    x = torch.randint(1, 100, [32, 128])
    glu = GateLinearUnit(embedding_size=300, num_filers=256, kernel_size=3, vocab_size=1000)
    out = glu(x)

