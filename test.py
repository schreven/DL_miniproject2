import torch
import math
from torch import FloatTensor as Tensor
from torch import LongTensor as TensorLong
import numpy as np

N = 1000

def generate_disc_set(N):
    input = Tensor(N, 2).uniform_(0, 1)
    target = input.pow(2).sum(1).sub(1/math.sqrt(2*math.pi)).sign().add(1).div(2).long()
    return input, target

train_input, train_target = generate_disc_set(N)
test_input, test_target = generate_disc_set(N)
