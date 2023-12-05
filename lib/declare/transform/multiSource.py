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
    return BaseTrans.tsfmSet(transforms=[
        msTrans.Separator(),
        BaseTrans.randomTrigger(
            transform=msTrans.Invalidator(),
            prob=0.2,
        ),
        msTrans.MSIterator(trans=singleSource.aug1())
    ])
