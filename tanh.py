import torch
import math
from torch import FloatTensor as Tensor
from torch import LongTensor as TensorLong
import numpy as np
from module import Module
import utils

class ReLU(Module):
    def __init__(self):
        super(ReLU, self).__init__():
        self.threshold = 0
        self.value = 0

    def forward(self, input):
        return tanh(input)

    def param(self):
        return []
