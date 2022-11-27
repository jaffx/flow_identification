import time

import torch


class flowHilbertTransform():
    n = 0

    def __init__(self, n):
        self.n = n


class toTensor():

    def __call__(self, data):
        return torch.Tensor(data)
