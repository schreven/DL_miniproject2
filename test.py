import torch
import math
from torch import FloatTensor as Tensor
from torch import LongTensor as TensorLong
import numpy as np
from network import Network
from sequential import Sequential
from linear import Linear
from relu import Relu
from tanh import Tanh
from mseloss import MSELoss
from utils import *

N = 1000

def generate_disc_set(N):
    input = Tensor(N, 2).uniform_(0, 1)
    target = input.pow(2).sum(1).sub(1/math.sqrt(2*math.pi)).sign().add(1).div(2).long()
    return input, target

train_input, train_target = generate_disc_set(N)
test_input, test_target = generate_disc_set(N)

train_input = train_input * 0.9
test_input = test_input * 0.9

nb_hidden = 25
nb_classes = 2

nb_train_samples = train_input.size(0)

eta = 1e-3 / nb_train_samples

# Possibility to add Linear, Tanh and ReLU
ann = Sequential(
                    Linear(2,nb_hidden),
                    Tanh(),
                    Linear(nb_hidden,nb_hidden),
                    Tanh(),
                    Linear(nb_hidden,nb_hidden),
                    Tanh(),
                    Linear(nb_hidden,2),
                    Tanh()
                )
MSE = MSELoss()

for k in range(0, 1000):

    acc_loss = 0
    nb_train_errors = 0

    ann.reset_parameters()

    for n in range(0, train_input.size(0)):
        output = ann.forward(train_input[n])
        pred = output.max(0)[1][0]

        if pred != train_target[n]: nb_train_errors += 1
        acc_loss = acc_loss + MSE.forward(output, train_target[n])
        ann.backward(MSE.backward(train_target[n],output))

    # Gradient step
    ann.update_parameters(eta)

    nb_test_errors = 0

    for n in range(0, test_input.size(0)):
        output_test = ann.forward(test_input[n])
        pred_test = output_test.max(0)[1][0]
        if pred_test != train_target[n]: nb_test_errors += 1

    print('{:d} acc_train_loss {:.02f} acc_train_error {:.02f}% test_error {:.02f}%'
          .format(k,
                  acc_loss,
                  (100 * nb_train_errors) / train_input.size(0),
                  (100 * nb_test_errors) / test_input.size(0)))
