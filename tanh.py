import torch
import math
from torch import FloatTensor as Tensor
from torch import LongTensor as TensorLong
import numpy as np
from module import Module
from utils import *
import utils

class Tanh(Module):
    def __init__(self):
        super(Tanh, self).__init__()
        # TODO Probably remove
        self.threshold = 0
        self.value = 0

    def forward(self, input):
        return tanh(input)

    def param(self):
        return []
