import torch
import math
from torch import FloatTensor as Tensor
from torch import LongTensor as TensorLong
import numpy as np
from network import Network

N = 1000

def generate_disc_set(N):
    input = Tensor(N, 2).uniform_(0, 1)
    target = input.pow(2).sum(1).sub(1/math.sqrt(2*math.pi)).sign().add(1).div(2).long()
    return input, target

train_input, train_target = generate_disc_set(N)
test_input, test_target = generate_disc_set(N)

nb_hidden = 25

nb_classes = 2
nb_train_samples = train_input.size(0)

eta = 1e-1 / nb_train_samples
epsilon = 1e-6

ann = Network(2,2,25)

for k in range(0, 1):

    for n in range(0, nb_train_samples):
        output = ann.forward(train_input[n])
        print(output)
