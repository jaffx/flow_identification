import sys

sys.path.append(".")
from model.Dataset import MSDataset
from model.DataLoader import DataLoader
from lib.declare import transform
from model.net import MHNet

path = "/Users/lyn/codes/python/Flow_Identification/Dataset/mv1/val"

dataset = MSDataset.MSDataset(length=2048, step=1024, path=path)
dataLoader = DataLoader.flowDataLoader(batch_size=10, showInfo=False, dataset=dataset,
                                       transform=transform.function.getTransform("ms-normalization"))
datas, labels, paths = dataLoader.getData()

net = MHNet.MHNet()
r = net(datas)
print(r)

