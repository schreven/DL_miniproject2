import torch
import math
from torch import FloatTensor as Tensor
from torch import LongTensor as TensorLong
import numpy as np
from module import Module

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Tensor(out_features, in_features)
        self.dweight = Tensor(self.weight.size())
        if bias:
            self.bias = Tensor(out_features)
        else:
            self.bias = None
        self.dbias = Tensor(self.bias.size())
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.uniform_(-stdv, stdv)

    def forward(self, input):
        if input.dim() == 2 and bias is not None:
            return torch.addmm(self.bias, input, self.weight.t())

        # mv for 1D vector
        output = torch.mv(self.weight, input)

        if self.bias is not None:
            output += self.bias
        return output
