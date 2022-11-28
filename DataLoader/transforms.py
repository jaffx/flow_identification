import time
from .Hilbert import getHilbert, HilbertBuild2

import torch


class flowHilbertTransform():

    def __init__(self, n):
        self.n = n
        self.Hilbert = getHilbert(n)

    def __call__(self, datas):
        ret = []
        for data in datas:
            ret.append([HilbertBuild2(data[0], self.Hilbert, self.n)])
        return torch.Tensor(ret)

class toTensor():

    def __call__(self, data):
        return torch.Tensor(data)



