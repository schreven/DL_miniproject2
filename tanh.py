import torch
import math
from torch import FloatTensor as Tensor
from torch import LongTensor as TensorLong
import numpy as np
from module import Module
from utils import *

class Tanh(Module):
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, input):
        return input.tanh()

    def backward(self, input):
        return input

    def update_parameters(self, eta):
        return

    def reset_gradient(self):
        return

    def param(self):
        return []
