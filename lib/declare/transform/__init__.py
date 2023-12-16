from model.transform import BaseTrans
from model.transform import Preprocess
from model.transform import DataAugmentation
from model.transform import multiSource as msTrans
from . import singleSource, function, multiSource

# 单数据源
function.addTransform("normalization", singleSource.normalization, "标准化")
function.addTransform("aug0.2", singleSource.aug1, "概率为0.2的小强度数据增强")
function.addTransform("aug0.3", singleSource.aug2, "概率为0.3的小强度数据增强")
function.addTransform("aug0.4", singleSource.aug3, "概率为0.4的小强度数据增强")

# 多数据源
function.addTransform("ms-normalization", multiSource.normalization, "标准化")
function.addTransform("ms-aug0.2", multiSource.MSAug1, "概率为0.2的小强度数据增强")
function.addTransform("ms-aug0.3", multiSource.MSAug2, "概率为0.2的小强度数据增强")
function.addTransform("ms-aug0.4", multiSource.MSAug3, "概率为0.2的小强度数据增强")
function.addTransform("ms-invalidator-20%", multiSource.MSInvalidator20(), "概率为0.2的数据源失活的数据增强")
function.addTransform("ms-invalidator-30%", multiSource.MSInvalidator30(), "概率为0.3的数据源失活的数据增强")
function.addTransform("ms-invalidator-40%", multiSource.MSInvalidator40(), "概率为0.4的数据源失活的数据增强")
function.addTransform("ms-invalidator-50%", multiSource.MSInvalidator50(), "概率为0.5的数据源失活的数据增强")
