import time

import numpy as np
import torch
import random


class xTransformException(Exception):
    Error_NUM = 0

    def __init__(self, message):
        super(xTransformException, self).__init__()
        self.message = message
        self.Error_NUM += 1

    def __str__(self):
        return f"xTransform error NO.{self.Error_NUM}:{self.message}"


class transformBase:
    def __init__(self):
        pass

    def __call__(self, x):
        return x

    def __str__(self):
        return f"{self.__class__.__name__}()"

    def str(self):
        return self.__str__()


class randomTrigger(transformBase):
    """
    设置一个概率p和一个transform，有概率p触发这个transform
    """

    def __init__(self, transform, prob=0.5):
        super().__init__()
        self.transform = transform
        self.prob = prob

    def __call__(self, x):
        trig = random.random() < self.prob
        return self.transform(x) if trig else x

    def __str__(self):
        ret = f"{self.__class__.__name__}(prob = {self.prob}, transform={str(self.transform)})"
        return ret


class randomSelector(transformBase):
    """
    随机的选择一个transform分支
    """

    def __init__(self, transforms: list):
        super().__init__()
        self.transforms = transforms

    def __call__(self, x):
        trans = random.choice(self.transforms)
        return trans(x)

    def __str__(self):
        ret = f"{self.__class__.__name__}( transforms=[ "
        for trans in self.transforms:
            ret += str(trans) + ", "
        ret += "])"
        return ret


class tsfmSelector(transformBase):
    """
    :brief
        顺序的选择一个transform分支
    """

    def __init__(self, transforms: list):
        super().__init__()
        self.transforms = transforms
        self.select = 0

    def __call__(self, x):
        trans = self.transforms[self.select]
        self.select = (self.select + 1) % len(self.transforms)
        return trans(x)

    def __str__(self):
        ret = f"{self.__class__.__name__}( transforms=[ "
        for trans in self.transforms:
            ret += str(trans) + ", "
        ret += "])"
        return ret


class tsfmSet(transformBase):
    # 递归经过所有transform
    def __init__(self, transforms: list):
        super().__init__()
        self.transforms = transforms

    def __call__(self, x):
        out = x
        for trans in self.transforms:
            out = trans(out)
        return out

    def __str__(self):
        ret = f"{self.__class__.__name__}( transforms=[ "
        for trans in self.transforms:
            ret += str(trans) + ", "
        ret += "])"
        return ret


class toTensor(transformBase):
    # 将输入转化为Tensor类型
    def __call__(self, data):
        return torch.Tensor(data)


class idtMapping(transformBase):
    # 恒等映射
    def __call__(self, x):
        return x
