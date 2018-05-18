import torch
import math
from torch import FloatTensor as Tensor
from torch import LongTensor as TensorLong
import numpy as np
from module import Module

class MSELoss(Module):
    def __init__(self):
        # TODO remove this
        self.test = []

    def forward(self, v, t):
        return (v - t).pow(2).sum()

    def backward(self, v, t):
        return 2 * (v - t)

    def parameters(self):
        return []
