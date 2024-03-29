import numpy as np
import torch
import math
from torch import FloatTensor as Tensor
from torch import LongTensor as TensorLong

def relu(x):
    out = Tensor(x.size(0))
    for ind, val in enumerate(x):
        out[ind] = float(max(0,float(val)))
    return out

def tanh(x):
    return x.tanh()

def dtanh(x):
    return 4 * (x.exp() + x.mul(-1).exp()).pow(-2)

def drelu(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x
