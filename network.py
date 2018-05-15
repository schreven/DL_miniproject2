import torch
import math
from linear import Linear
from torch import FloatTensor as Tensor
from torch import LongTensor as TensorLong
from module import Module

class Network(Module):
    def __init__(self, nb_in, nb_out, nb_hidden):
        super(Network, self).__init__()
        self.fc1 = Linear(nb_in, nb_hidden)
        self.fc2 = Linear(nb_hidden, nb_hidden)
        self.fc3 = Linear(nb_hidden, nb_out)

    def forward(self, x, w, b):
        x = self.fc1.forward(x,w,b)
        return x
