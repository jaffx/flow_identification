import time
from .Hilbert import getHilbert, HilbertBuild2

import torch
import numpy as np
from numpy import fft


class xTransformException(Exception):
    Error_NUM = 0

    def __init__(self, message):
        super(xTransformException, self).__init__()
        self.message = message
        self.Error_NUM += 1

    def __str__(self):
        return f"xTransform error NO.{self.Error_NUM}:{self.message}"


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


class spaciousFolder():
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, datas):
        ret = []
        for data in datas:
            data = data[0]
            if len(data) != self.height * self.width:
                raise xTransformException(f"SpaciousFolder only accept datas at length at {self.height * self.width} ")
            data = [[data[i * self.width:(i + 1) * self.width] for i in range(self.height)]]
            ret.append(data)
        return torch.Tensor(ret)


class FFT_Transform():
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, datas):
        ret = []
        for data in datas:
            data = data[0]
            if len(data) != self.height * self.width:
                raise xTransformException(f"FFT_Transform only accept datas at length at {self.height * self.width} ")
            data = [data[i * self.width:(i + 1) * self.width] for i in range(self.height)]
            rdata = []
            for d in data:
                ff_d = abs(np.fft.fft(d))
                ff_d = ff_d / self.width
                rdata.append(list(ff_d))
            ret.append([rdata])
        ret = torch.Tensor(ret)
        return ret
