import torch
import math
from torch import FloatTensor as Tensor
from torch import LongTensor as TensorLong
import numpy as np
from module import Module

class MSELoss(Module):
    def __init__(self):
        # TODO remove this
        self.not = []

    def forward(self, input, target):
        return (v - t).pow(2).sum()

    def dloss(self, input, target):
        return 2 * (v - t)
