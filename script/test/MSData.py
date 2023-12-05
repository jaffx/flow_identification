import sys

sys.path.append(".")
from model.Dataset import MSDataset
from model.DataLoader import DataLoader
from lib.declare import transform
from model.net import MHNet

path = "/Users/lyn/codes/python/Flow_Identification/Dataset/mv1/val"

dataset = MSDataset.MSDataset(length=10, step=20, path=path)
dataLoader = DataLoader.flowDataLoader(batch_size=10, showInfo=False, dataset=dataset,
                                       transform=transform.function.getTransform("ms-invalidator-normalization"))
for i in range(10):
    datas, labels, paths = dataLoader.getData()
    print(datas)

# net = MHNet.MHNet()
# r = net(datas)
# print(r)
#
