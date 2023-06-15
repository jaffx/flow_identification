import os
import yaml


def getDeviceName():
    if os.path.isdir("/hy-tmp"):
        return "hy"
    return "mac"


def getDatasetPath(dataset="wms_old", device="mac"):
    with open("conf/dataset_path.yaml") as fp:
        datasets = yaml.full_load(fp)
        fp.close()
    assert dataset in datasets and device in datasets[dataset], "数据集配置不存在"
    return datasets[dataset][device]
