import torch
import math
from torch import FloatTensor as Tensor
from torch import LongTensor as TensorLong
#from parameter import Parameter
import numpy as np
from module import Module

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Tensor(out_features, in_features)
        if bias:
            self.bias = Tensor(out_features)
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.uniform_(-stdv, stdv)

    def forward(self, input, weight, bias):

        if input.dim() == 2 and bias is not None:
            # fused op is marginally faster
            return torch.addmm(bias, input, weight.t())
        # mv for 1D vector
        output = torch.mv(weight, input)

        if bias is not None:
            output += bias
        return output
