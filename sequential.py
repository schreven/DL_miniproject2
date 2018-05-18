import torch
import math
from torch import FloatTensor as Tensor
from torch import LongTensor as TensorLong
import numpy as np
from module import Module
from relu import Relu
from linear import Linear
from tanh import Tanh

class Sequential(Module):
    def __init__(self, *args):
        super(Sequential, self).__init__()
        self.modules = []
        for mod in args:
            self.modules.append(mod)

    def forward(self, input):
        x = input
        for mod in self.modules:
            x = mod.forward(x)
        return x
