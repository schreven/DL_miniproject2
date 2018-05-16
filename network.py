import torch
import math
from linear import Linear
from torch import FloatTensor as Tensor
from torch import LongTensor as TensorLong
from module import Module
from utils import *

class Network(Module):
    def __init__(self, nb_in, nb_out, nb_hidden):
        super(Network, self).__init__()
        self.fc1 = Linear(nb_in, nb_hidden)
        self.fc2 = Linear(nb_hidden, nb_out)

    def forward(self, x):
        x = relu(self.fc1.forward(x))
        x = relu(self.fc2.forward(x))
        return x
