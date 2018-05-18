import torch
import math
from torch import FloatTensor as Tensor
from torch import LongTensor as TensorLong
import numpy as np
from module import Module
from utils import *

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
        self.previous_input = Tensor()
        self.current_output = Tensor()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.uniform_(-stdv, stdv)

    def reset_gradient(self):
        self.dweight.zero_()
        self.dbias.zero_()

    def forward(self, input):
        self.previous_input = input
        #if input.dim() == 2 and bias is not None:
        #    return torch.addmm(self.bias, input, self.weight.t())

        # mv for 1D vector
        #output = torch.mv(self.weight, input)
        output = self.weight.mv(input)
        if self.bias is not None:
            output += self.bias
        self.current_output = output
        return output

    def backward(self, input):
        dl_ds = dtanh(self.current_output) * input
        dl_dx = self.weight.t().mv(dl_ds)

        self.dweight.add_(dl_ds.view(-1, 1).mm(self.previous_input.view(1, -1)))
        self.dbias.add_(dl_ds)

        return dl_dx

    def update_parameters(self, eta):
        self.weight = self.weight - eta * self.dweight
        self.bias = self.bias - eta * self.dbias

    def parameters(self):
        return self.weight, self.bias
