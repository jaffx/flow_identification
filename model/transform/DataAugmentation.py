import random
import sys
from . import BaseTrans as BT

import numpy as np


class normalizedRandomNoise(BT.transformBase):
    # 添加一个正态分布噪声
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def __call__(self, x=None):
        shape = x.shape
        noise = np.random.normal(self.mean, self.std, shape)
        return x + noise if x is not None else noise

    def __str__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class randomNoise(BT.transformBase):
    """
    为输入数据增加一个随机噪声，噪声符合均匀分布，范围为(noise_min, noise_max)
    """

    def __init__(self, noise_min, noise_max):
        super().__init__()
        self.noise_min, self.noise_max = noise_min, noise_max

    def __call__(self, x=None):
        shape = x.shape
        noise = np.random.random(size=shape) * (self.noise_max - self.noise_min) + self.noise_min
        return x + noise if x is not None else noise

    def __str__(self):
        return f"{self.__class__.__name__}(noise_min={self.noise_min}, noise_max={self.noise_max})"


class dropout(BT.transformBase):
    """
    数据点按照比例随机设置为0
    """

    def __init__(self, rate=0.5):
        super().__init__()
        self.rate = rate

    def __call__(self, x: np.array):
        for item in x:
            shape: tuple = item.shape
            mask = np.random.normal(loc=0.5, scale=0.5, size=shape)
            mask[mask > self.rate] = 1
            mask[mask < self.rate] = 0
            item *= mask
        return x

    def __str__(self):
        return f"{self.__class__.__name__}(rate={self.rate})"


class randomRangeMasking(BT.transformBase):
    """
    @description
        随机区间数据遮蔽
        对输入数据的一个区间进行遮蔽，将其值转化为0
        数据输入只能包含三个维度(batch, channel, length)
    """

    def __init__(self, rate=0.5):
        """
        :param rate 遮蔽区间的长度比例
        """
        super().__init__()
        self.rate = rate

    def __call__(self, x):
        assert len(x.shape) == 3, \
            f"随机数据遮蔽只接受数据维度为(batch, channel, length)的数据,当前输入数据维度为{x.shape}"
        batch, channel, length = x.shape
        mask_len = int(length * self.rate)
        for i in range(batch):
            for j in range(channel):
                start: int = random.randint(0, length - mask_len)
                x[i, j, start:start + mask_len] = np.zeros(shape=mask_len)
        return x

    def __str__(self):
        return f"{self.__class__.__name__}(rate={self.rate})"


class dataDeath(BT.transformBase):
    """
    数据死亡，数据转化为全0
    """

    def __call__(self, x):
        return np.zeros_like(x)
