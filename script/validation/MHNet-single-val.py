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
from model.analyzer import Analyzer as aly
from lib import xyq

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


class validator:
    def __init__(self, path):
        self.infos = None
        self.path = path
        netName = self.getBasicInfo("Net")
        self.net = xyq.function.getNet(netName, num_classes=7)
        weightPath = os.path.join(self.path, "weight.pth")
        self.net.load_state_dict(
            torch.load(weightPath, map_location=device))

    def writeValRecord(self, info):
        valPath = os.path.join(self.path, "val.record")
        with open(valPath, "a") as fp:
            s = json.dumps(info)
            fp.write(s + "\n")
            fp.close()

    def getBasicInfo(self, key):
        if self.infos is None:
            infoPath = os.path.join(self.path, "info.yaml")
            with open(infoPath) as fp:
                self.infos = yaml.safe_load(fp)
        return self.infos[key]

    def valWMS(self):
        transformName = "normalization"
        valPath = os.path.join(datasetPath, "val", "wms")
        valDataset = Dataset.flowDataset(path=valPath, length=dataLength, step=step, name="validation")
        tsfm = transform.function.getTransform(transformName)
        valLoader = DataLoader.flowDataLoader(dataset=valDataset, transform=tsfm,
                                              batch_size=batchSize, showInfo=True)
        acc, count = 0, 0
        with torch.no_grad():
            while valLoader.isReadable():
                data, label, path = valLoader.getData()
                if isinstance(data, torch.Tensor):
                    data = data.to(device)
                else:
                    for i in range(len(data)):
                        data[i] = data[i].to(device)
                label = torch.Tensor(label).to(device)
                predict_y = self.net.callNet1(data)
                predict_label = torch.argmax(predict_y, dim=1)
                for i in range(len(predict_label)):
                    prl, tl = int(predict_label[i]), int(label[i])
                    if prl == tl:
                        acc += 1
                    count += 1
        return acc / count

    def valPressure(self, ):
        transformName = "normalization"
        valPath = os.path.join(datasetPath, "val", "pressure")
        valDataset = Dataset.flowDataset(path=valPath, length=dataLength, step=step, name="validation")
        tsfm = transform.function.getTransform(transformName)
        valLoader = DataLoader.flowDataLoader(dataset=valDataset, transform=tsfm,
                                              batch_size=batchSize, showInfo=True)
        acc, count = 0, 0
        with torch.no_grad():
            while valLoader.isReadable():
                data, label, path = valLoader.getData()
                if isinstance(data, torch.Tensor):
                    data = data.to(device)
                else:
                    for i in range(len(data)):
                        data[i] = data[i].to(device)
                label = torch.Tensor(label).to(device)
                predict_y = self.net.callNet2(data)
                predict_label = torch.argmax(predict_y, dim=1)
                for i in range(len(predict_label)):
                    prl, tl = int(predict_label[i]), int(label[i])
                    if prl == tl:
                        acc += 1
                    count += 1
        return acc / count

    def valFusion(self, ):
        transformName = "ms-normalization"
        valPath = os.path.join(datasetPath, "val")
        valDataset = MSDataset.MSDataset(path=valPath, length=dataLength, step=step, name="ms-validation")
        tsfm = transform.function.getTransform(transformName)
        valLoader = DataLoader.flowDataLoader(dataset=valDataset, transform=tsfm,
                                              batch_size=batchSize, showInfo=True)
        acc, count = 0, 0
        with torch.no_grad():
            while valLoader.isReadable():
                data, label, path = valLoader.getData()
                if isinstance(data, torch.Tensor):
                    data = data.to(device)
                else:
                    for i in range(len(data)):
                        data[i] = data[i].to(device)
                label = torch.Tensor(label).to(device)
                predict_y = self.net.callFusion(data)
                predict_label = torch.argmax(predict_y, dim=1)
                for i in range(len(predict_label)):
                    prl, tl = int(predict_label[i]), int(label[i])
                    if prl == tl:
                        acc += 1
                    count += 1
        return acc / count

    def val(self, ):
        accWMS = self.valWMS()
        print(f"WMS:{accWMS:.4f}")
        self.writeValRecord({"accWMS": accWMS})

        accPressure = self.valPressure()
        print(f"Pressure:{accPressure:.4f}")
        self.writeValRecord({"accPressure": accPressure})

        accFusion = self.valFusion()
        print(f"Fusion:{accFusion:.4f}")
        self.writeValRecord({"accFusion": accFusion})


def val(path):
    validator(path).val()


resultDir = "/Users/lyn/codes/python/Flow_Identification/Flow_Identification/ex_result/复用方式和数据源失活比例实验"
results = os.listdir(resultDir)
for r in results:
    resultPath = os.path.join(resultDir, r)
    val(resultPath)
