"""
Scanner，整体视角管理训练结果
"""
from . import *
import os


def _ex(value):
    return value is not None


def _gt(value1, value2):
    try:
        value1 = float(value1)
        value2 = float(value2)
    except Exception:
        return None
    return value1 > value2


def _is(value1, value2):
    try:
        if isinstance(value2, int) or isinstance(value2, float):
            value1 = float(value1)
            value2 = float(value2)
            return value1 == value2
        return value1 == value2
    except Exception:
        return None


def isSatisfied(value1, op, value2):
    if op == "ex":
        return _ex(value1)
    if op == "gt":
        return _gt(value1, value2)
    if op == "lt":
        return not _gt(value1, value2)
    if op == "is" or op == "eq":
        return _is(value1, value2)


class Scanner:
    __valid_op__ = {"ex", "gt", "lt", "eq", "is", "in"}

    def __init__(self, path: str):
        self.path: str = path
        self.alyList: list = []

    def load(self):
        assert os.path.isdir(self.path), f"Scanner 路径【{self.path}】不存在"
        results = os.listdir(self.path)
        for r in results:
            alyer = Analyzer.Analyzer(path=os.path.join(self.path, r))
            # 判断文件格式是否完整
            if not alyer.checkResult():
                continue
            self.alyList.append(alyer)

    def filter(self, attr, op, value=None):
        assert op in self.__valid_op__
        res = Scanner(self.path)
        for aly in self.alyList:
            info = aly.getInfo(attr)
            if info is None:
                continue
            if isSatisfied(info, op, value):
                res.alyList.append(aly)
        return res
