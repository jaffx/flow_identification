from . import *

"""
定义单数据源预处理使用的transform
在__init__.py文件中注册到transform列表中
"""


def normalization():
    return BaseTrans.tsfmSet([
        Preprocess.normalization(),
        BaseTrans.toTensor()
    ])


def aug1():
    return BaseTrans.tsfmSet([
        Preprocess.normalization(),
        BaseTrans.randomTrigger(
            BaseTrans.randomSelector([
                DataAugmentation.randomRangeMasking(0.1),
                DataAugmentation.randomNoise(-0.05, 0.05),
                DataAugmentation.normalizedRandomNoise(0, 0.05),
            ]),
            prob=0.2
        ),
        BaseTrans.toTensor()
    ])


def aug2():
    BaseTrans.tsfmSet([
        Preprocess.normalization(),
        BaseTrans.randomTrigger(
            BaseTrans.randomSelector([
                DataAugmentation.randomRangeMasking(0.1),
                DataAugmentation.randomNoise(-0.05, 0.05),
                DataAugmentation.normalizedRandomNoise(0, 0.05),
            ]),
            prob=0.3
        ),
        BaseTrans.toTensor()
    ])


def aug3():
    return BaseTrans.tsfmSet([
        Preprocess.normalization(),
        BaseTrans.randomTrigger(
            BaseTrans.randomSelector([
                DataAugmentation.randomRangeMasking(0.1),
                DataAugmentation.randomNoise(-0.05, 0.05),
                DataAugmentation.normalizedRandomNoise(0, 0.05),
            ]),
            prob=0.4
        ),
        BaseTrans.toTensor()
    ])
