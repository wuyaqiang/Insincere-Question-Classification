#coding=utf-8
'''
Attention 结构
'''
import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, feature_dim, step_num, bias=True, **kwargs):
        super(Attention, self).__init__()

        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_num = step_num

        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_num))

    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_num = self.step_num

        eij = torch.mm(x.contiguous().view(-1, feature_dim), self.weight)   # (batch_size * step_num, 1)
        eij = eij.view(-1, step_num)    # (batch_size, step_num)

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)