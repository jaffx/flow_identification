import random

from . import *


class MSIterator(BaseTrans.transformBase):
    def __init__(self, trans: BaseTrans.transformBase):
        """
        :param trans: 传入一个transform,所有的数据循环通过该transform处理
        """
        super().__init__()
        self.trans = trans

    def __call__(self, data):
        ret = []
        for d in data:
            ret.append(self.trans(d) if d is not None else None)
        return ret


class Invalidator(BaseTrans.transformBase):
    """
    在数据源中选择一个置为None
    """

    def __call__(self, x):
        l: int = len(x)
        idx = random.randint(0, l - 1)
        x[idx] = None
        return x


class Separator(BaseTrans.transformBase):
    def __call__(self, x):
        ret = []
        for d in x:
            ret.append(d)
        return ret
