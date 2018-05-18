import torch
import math
from torch import FloatTensor as Tensor
from torch import LongTensor as TensorLong
import numpy as np
from module import Module
from utils import *

class Relu(Module):
    def __init__(self):
        super(Relu, self).__init__()

    def backward(self, input):
        return input

    def forward(self, input):
        return relu(input)

    def param(self):
        return []

    def update_parameters(self, eta):
        return

    def reset_gradient(self):
        return
