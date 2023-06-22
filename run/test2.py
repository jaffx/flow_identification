import time

from lib.transforms import BaseTrans as BT
from lib.transforms import Preprocess as PP
from lib.transforms import DataAugmentation as DA
from lib.Dataset import Dataset
from lib.DataLoader import DataLoader
from lib.xyq import x_time
import numpy as np


@x_time.showRuningTime
def test(dl):
    dl.getData()


def main():
    path1 = "/Users/lyn/codes/python/Flow_Identification/Dataset/v1/WMS_Simple_B/val"
    path2 = "/Users/lyn/codes/python/Flow_Identification/Dataset/v2/WMS/v2_WMS_Label_Simple_B/val"
    ds1 = Dataset.flowDataset(path1, length=4096, step=2048)
    ds2 = Dataset.flowDataset(path2, length=4096, step=2048)
    transform = BT.transfrom_set(
        [PP.normalization(),DA.random_range_masking(), BT.toTensor(), ]
    )
    dl1 = DataLoader.flowDataLoader(ds1, 64, transform, True)
    dl2 = DataLoader.flowDataLoader(ds2, 64, transform, True)

    test(dl1)
    test(dl2)


if __name__ == "__main__":
    main()
