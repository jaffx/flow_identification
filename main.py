import time

from DataLoader.Dataset import flowDataset
from xyq import x_printer as printer
from DataLoader.DataLoader import flowDataLoader
from DataLoader.transforms import toTensor
from xyq import x_time as xtime



@xtime.showRuningTime
def test2():
    train_set2 = flowDataset(path="../FlowDataset/Datas2/val", length=128 * 128, step=128 * 64, name="Val Set")
    train_set2.getDatasetInfo()
    to_tensor = toTensor()
    train_loader = flowDataLoader(train_set2, 8, to_tensor, showInfo=True)
    while train_loader.getReadable():
        train_loader.getData()


test2()

