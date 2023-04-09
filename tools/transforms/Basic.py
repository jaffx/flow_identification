import time
import torch


class xTransformException(Exception):
    Error_NUM = 0

    def __init__(self, message):
        super(xTransformException, self).__init__()
        self.message = message
        self.Error_NUM += 1

    def __str__(self):
        return f"xTransform error NO.{self.Error_NUM}:{self.message}"


class transfromsSet:
    def __init__(self, transform_list: list):
        self.transfroms = transform_list

    def __call__(self, x):
        out = x
        for trans in self.transfroms:
            out = trans(out)
        return out


class toTensor:

    def __call__(self, data):
        return torch.Tensor(data)
