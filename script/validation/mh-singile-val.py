"""
对MHNet进行单数据源预测
"""
import torch
import os

import yaml
import sys

sys.path.append(".")
import lib.xyq.printer as printer
import lib.xyq.format as formatter
from model.net import MHNet
from lib.declare import transform
from model.Dataset import MSDataset, Dataset
from model.DataLoader import DataLoader
from model.analyzer import analyzer as aly
from lib import xyq

# 权重文件路径
weightPath = "/Users/lyn/codes/python/Flow_Identification/Flow_Identification/ex_result/train/20231216.111609_MHNet/weight.pth"
# 预处理器名称

# 数据集名称
datasetName = "mv1"
# 环境名称
deviceName = xyq.function.getDeviceName()
# 数据长度,采样步长
dataLength = 4096
step = dataLength // 2

batchSize = 64

# 加载分析器


# 加载数据集
datasetPath, clsNum = xyq.function.getMSDatasetInfo(datasetName, deviceName)

transformName = "normalization"
valPath = os.path.join(datasetPath, "val", "wms")
valDataset = Dataset.flowDataset(path=valPath, length=dataLength, step=step, name="validation")
# transformName = "ms-normalization"
# valPath = os.path.join(datasetPath, "val")
# valDataset = MSDataset.MSDataset(path=valPath, length=dataLength, step=step, name="ms-validation")

transform = transform.function.getTransform(transformName)
valLoader = DataLoader.flowDataLoader(dataset=valDataset, transform=transform,
                                      batch_size=batchSize, showInfo=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = MHNet.MHNet()
net.load_state_dict(torch.load(weightPath, map_location=device))
acc, count = 0, 0
with torch.no_grad():
    while valLoader.getReadable():
        data, label, path = valLoader.getData()
        if isinstance(data, torch.Tensor):
            data = data.to(device)
        else:
            for i in range(len(data)):
                data[i] = data[i].to(device)
        label = torch.Tensor(label).to(device)
        # predict_y = net.callFusion(data)
        predict_y = net.callNet1(data)
        # predict_y = net.callNet2(data)
        predict_label = torch.argmax(predict_y, dim=1)
        for i in range(len(predict_label)):
            prl, tl = int(predict_label[i]), int(label[i])
            if prl == tl:
                acc += 1
            count += 1
print(acc / count)
