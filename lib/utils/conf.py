import os
import yaml
from lib.utils.transform import transform


def getDeviceName():
    if os.path.isdir("/hy-tmp"):
        return "hy"
    return "mac"


def getDatasetPath(dataset="wms_old", device="mac"):
    with open("conf/dataset_path.yaml") as fp:
        datasets = yaml.full_load(fp)
        fp.close()
    if dataset not in datasets or device not in datasets[dataset]:
        print(f"数据集配置不存在：{device}-{dataset},支持的数据集包括:")
        for key in datasets:
            print(f"\t{key}")
        exit(1)
    return datasets[dataset][device]


def getTransform(name):
    try:
        return transform.getTransform(name)
    except:
        infos = transform.getAllTransformInfos()
        print(f"指定的transform错误，请选择如下transform")
        for info in infos:
            print(f"\t{info['name']:<8}\t{info['desc']}")
        exit(1)
