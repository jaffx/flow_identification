import os
import yaml
from xlib.utils.transform import transform
from xlib.conf import config
from xlib.modifier.modepoch import ModifierEpoch as Modifier


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


def getDatasetPath(dataset="wms_old", device="mac"):
    """
    根据数据集名称和设备名称获取数据集路径
    :dataset str 数据集名
    :device str 设备名
    """
    with open("conf/dataset_path.yaml") as fp:
        datasets = yaml.full_load(fp)
        fp.close()
    if dataset not in datasets or device not in datasets[dataset]:
        print(f"数据集配置不存在：{device}-{dataset},支持的数据集包括:")
        for key in datasets:
            print(f"\t{key}")
        exit(1)
    return datasets[dataset][device]


def getDatasetInfo(dataset):
    """
    获取数据集的配置信息
    """
    _config = config.config("conf/dataset_info.yaml")
    _info = _config.get(dataset)
    if _info is None:
        datasets = _config.get("/")
        print(f"数据集配置【{dataset}】不存在,支持的数据集包括:")
        for key in datasets:
            print(f"\t{key}")
        exit(1)
    return _info


def getDatasetPathAndClassNum(dataset: str, device: str):
    """
    获取数据集路劲和classnum
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
        return transform.getTransform(name)
    except:
        infos = transform.getAllTransformInfos()
        print(f"指定的transform错误，请选择如下transform")
        for info in infos:
            print(f"\t{info['name']:<8}\t{info['desc']}")
        exit(1)


def getModifier(name: str):
    if name is None or name == "None":
        return None
    modifier = Modifier(name)
    return modifier
