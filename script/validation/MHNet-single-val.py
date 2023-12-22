"""
对MHNet进行单数据源预测
"""
import json

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
resPath = "/Users/lyn/codes/python/Flow_Identification/Flow_Identification/ex_result/数据源随机失活比例效果实验/20231216.171728_MHNet"
weightPath = os.path.join(resPath, "weight.pth")
valPath = os.path.join(resPath, "val.record")

# 数据集名称
datasetName = "mv1"
# 环境名称
deviceName = xyq.function.getDeviceName()
# 数据长度,采样步长
dataLength = 4096
step = dataLength // 2

batchSize = 64

datasetPath, clsNum = xyq.function.getMSDatasetInfo(datasetName, deviceName)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = MHNet.MHNet()
net.load_state_dict(torch.load(weightPath, map_location=device))


def writeValRecord(info):
    with open(valPath, "a") as fp:
        s = json.dumps(info)
        fp.write(s + "\n")
        fp.close()


def valWMS():
    transformName = "normalization"
    valPath = os.path.join(datasetPath, "val", "wms")
    valDataset = Dataset.flowDataset(path=valPath, length=dataLength, step=step, name="validation")
    tsfm = transform.function.getTransform(transformName)
    valLoader = DataLoader.flowDataLoader(dataset=valDataset, transform=tsfm,
                                          batch_size=batchSize, showInfo=True)
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
            predict_y = net.callNet1(data)
            predict_label = torch.argmax(predict_y, dim=1)
            for i in range(len(predict_label)):
                prl, tl = int(predict_label[i]), int(label[i])
                if prl == tl:
                    acc += 1
                count += 1
    return acc / count


def valPressure():
    transformName = "normalization"
    valPath = os.path.join(datasetPath, "val", "pressure")
    valDataset = Dataset.flowDataset(path=valPath, length=dataLength, step=step, name="validation")
    tsfm = transform.function.getTransform(transformName)
    valLoader = DataLoader.flowDataLoader(dataset=valDataset, transform=tsfm,
                                          batch_size=batchSize, showInfo=True)
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
            predict_y = net.callNet2(data)
            predict_label = torch.argmax(predict_y, dim=1)
            for i in range(len(predict_label)):
                prl, tl = int(predict_label[i]), int(label[i])
                if prl == tl:
                    acc += 1
                count += 1
    return acc / count


def valFusion():
    transformName = "ms-normalization"
    valPath = os.path.join(datasetPath, "val")
    valDataset = MSDataset.MSDataset(path=valPath, length=dataLength, step=step, name="ms-validation")
    tsfm = transform.function.getTransform(transformName)
    valLoader = DataLoader.flowDataLoader(dataset=valDataset, transform=tsfm,
                                          batch_size=batchSize, showInfo=True)
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
            predict_y = net.callFusion(data)
            predict_label = torch.argmax(predict_y, dim=1)
            for i in range(len(predict_label)):
                prl, tl = int(predict_label[i]), int(label[i])
                if prl == tl:
                    acc += 1
                count += 1
    return acc / count


accWMS = valWMS()
print(f"WMS:{accWMS:.4f}")
writeValRecord({"accWMS": accWMS})

accPressure = valPressure()
print(f"Pressure:{accPressure:.4f}")
writeValRecord({"accPressure": accPressure})

accFusion = valFusion()
print(f"Fusion:{accFusion:.4f}")
writeValRecord({"accFusion": accFusion})
