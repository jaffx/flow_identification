from . import *


def normalization():
    return msTrans.MSIterator(trans=singleSource.normalization())


def MSAug1():
    return msTrans.MSIterator(trans=singleSource.aug1())


def MSAug2():
    return msTrans.MSIterator(trans=singleSource.aug2())


def MSAug3():
    return msTrans.MSIterator(trans=singleSource.aug3())


def MSInvalidatorNormalization():
    """
    单数据源0.2概率失活
    """
    return BaseTrans.tsfmSet(transforms=[
        msTrans.Separator(),
        BaseTrans.randomTrigger(
            transform=msTrans.Invalidator(),
            prob=0.2,
        ),
        msTrans.MSIterator(trans=singleSource.normalization())
    ])


def MSInvalidatorFullNormalization():
    """
    每次必然会使得一个数据源失活
    """
    return BaseTrans.tsfmSet(transforms=[
        msTrans.Separator(),
        BaseTrans.randomTrigger(
            transform=msTrans.Invalidator(),
            prob=1,
        ),
        msTrans.MSIterator(trans=singleSource.normalization())
    ])
