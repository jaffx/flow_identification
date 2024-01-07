import functools
import os
import yaml
from ..declare import transform as dclrTransform, net as dclrNet
from model.config import config
from model.modifier.epoch import ModifierEpoch as Modifier
from . import *


def getDeviceName():
    """
    :brief 获取当前设备的名称
    :return 设备名
        - hy    恒源云
        - mac   lyn 的 Macbook Pro M2
    """
    if os.path.isdir("/hy-tmp"):
        return "hy"
    return "mac"


@functools.lru_cache(maxsize=5)
def getDatasetInfo(dataset):
    """
    获取数据集的配置信息
    """
    _config = config.config(const.XYQ_CONF_PATH_DATASET_INFO)
    _info = _config.get(dataset)
    if _info is None:
        datasets = _config.get("/")
        printer.xprint_red(f"数据集配置【{dataset}】不存在,支持的数据集包括:")
        for key in datasets:
            printer.xprint_red(f"\t{key}")
        raise Exception("Dataset Not Found")
    return _info


def getDatasetPath(dataset, device="mac"):
    """
    根据数据集名称和设备名称获取数据集路径
    :dataset str 数据集名
    :device str 设备名
    """
    _info = getDatasetInfo(dataset)
    assert "paths" in _info, f"数据集【{dataset}】配置信息错误"
    assert device in _info["paths"], f"数据集【{dataset}】在设备【{device}】上路径未找到"
    return _info["paths"][device]


def getDatasetPathAndClassNum(dataset: str, device: str):
    """
    获取数据集路径和class_num
    :return tuple 路径，分类数量
    """
    _info = getDatasetInfo(dataset)
    assert "class_num" in _info and "paths" in _info, f"数据集【{dataset}】配置信息错误"
    assert device in _info["paths"], f"数据集【{dataset}】在设备【{device}】上路径未找到"
    return _info["paths"][device], _info["class_num"]


def getMSDatasetInfo(dataset: str, device: str):
    """
    获取多源数据集的基本信息，路径、class num、数据源列表
    :return 路径，分类数量
    """
    _info = getDatasetInfo(dataset)
    assert "class_num" in _info and "paths" in _info, f"数据集【{dataset}】配置信息错误"
    assert device in _info["paths"], f"数据集【{dataset}】在设备【{device}】上路径未找到"
    return _info["paths"][device], _info["class_num"]


def getTransform(name):
    """
    获取转换器
    """
    try:
        return dclrTransform.function.getTransform(name)
    except:
        infos = dclrTransform.function.getAllTransformInfos()
        printer.xprint_red(f"指定的transform错误，请选择如下transform")
        for info in infos:
            printer.xprint_red(f"\t{info['name']:<8}\t{info['desc']}")
        raise Exception("transform not found")


def getModifier(name: str):
    if name is None or name == "None":
        return None
    modifier = Modifier(name)
    return modifier


def getNet(name: str, **kwargs):
    if name is None or name == "None":
        return None
    try:
        netCreator = dclrNet.function.getNet(name)
        net = netCreator(**kwargs)
        return net
    except AssertionError:
        infos = dclrNet.function.getAllNetInfo()
        printer.xprint_red(f"神经网络模型「{name}」指定错误，请选择如下网络：")
        for info in infos:
            printer.xprint_red(f"\t{info['name']:<8}\t{info['desc']}")
        raise Exception("Net not found")
    except Exception as e:
        printer.xprint_red(f"创建网络出现错误：{e}")
        raise Exception(f"创建网络错误:{e}")
