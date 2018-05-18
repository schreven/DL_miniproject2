import torch
import math
from torch import FloatTensor as Tensor
from torch import LongTensor as TensorLong
import numpy as np
from module import Module

class Sequential(Module):
    def __init__(self, *args):
        self.modules = []
        for mod in args:
            self.modules.append(mod)

    def forward(self, input):
        for mod in self.modules:
            x = mod.forward(x)
        return x


    def add_module(mod):
        self.modules.append(mod)
