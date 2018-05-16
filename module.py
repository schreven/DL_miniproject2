import torch

# Abstract class (methods have to be redefined)
class Module(object):

    def __init__(self):
        self.test = []

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def parameters(self):
        return []