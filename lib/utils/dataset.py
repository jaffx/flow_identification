import os
import yaml
import os

def get_dataset_path(dataset="wms_old", device="mac"):
    with open("conf/dataset_path.yaml") as fp:
        datasets = yaml.full_load(fp)
        fp.close()
    return datasets[dataset][device]

