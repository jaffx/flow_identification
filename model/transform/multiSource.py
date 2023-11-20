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
            ret.append(self.trans(d))
        return ret

