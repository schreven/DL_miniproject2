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
    out = Tensor(x.size(0))
    out = np.tanh(x)
    return out
