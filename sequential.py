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
        for ind, mod in enumerate(self.modules):
            x = mod.forward(x)
        return x

    def backward(self, output):
        x = output
        for mod in reversed(self.modules):
            x = mod.backward(x)
        return x

    def update_parameters(self, eta):
        for mod in self.modules:
            mod.update_parameters(eta)
        return

    def reset_parameters(self):
        for mod in self.modules:
            mod.reset_gradient()
        return
