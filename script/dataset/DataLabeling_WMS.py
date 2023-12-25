"""
@date 2023年06月04日
@author xuyongqi
加载一个网络模型，读取WMS数据，进行识别并将识别结果保存到result.json文件中
"""

import json
import os
import shutil
import sys

sys.path.append("../v2data_process")
import torch
from xlib.transforms import Preprocess as PP
from xlib.transforms import BaseTrans as BT
from xlib.DataLoader import DataLoader as DL
from xlib.Dataset import Dataset
from xlib.xyq import x_printer as XP
from model.Res1D import resnet1d34
import re
sys.path.append("../v2data_process")
from model.Res1D import resnet1d34
import csv

class_num = 7


def dealResultJson(path):
    result = json.load(open(path))
    pattern = re.compile(".*G(\d+)L(\d+).*")
    rows = []
    for file in result:
        props = result[file]
        sumProp = sum(props)
        for i in range(len(props)):
            props[i] = round(props[i] / sumProp,2)

        maxProp = max(props)
        maxPropIdx = 0
        for i in range(len(props)):
            if props[i] == maxProp:
                maxPropIdx = i
        fileName = file.split("/")
        Dir = fileName[-2]
        fileName = fileName[-1]
        ret = pattern.match(fileName)
        gas = ret.group(1)
        liquid = ret.group(2)
        row = [Dir,fileName, int(gas), int(liquid), props[0], props[1], props[2], props[3], props[4], props[5], props[6], maxPropIdx, maxProp]
        rows.append(row)
    filename = path.split("/")[-1].split(".")[0]
    fp = open(f"{filename}.csv", "w+")
    writer = csv.writer(fp, "excel")
    headers = ['文件夹','文件名','气速','液速', '0', '1', '2', '3', '4', '5', '6', '模型标签', '置信度']
    writer.writerow(headers)
    writer.writerows(rows)
    fp.close()


def do_process(path, model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    originPath = path
    modelPath = model
    train_trans = BT.transform_set([
        PP.normalization(),
        BT.toTensor()
    ])

    # fDataList = []
    # files = os.listdir(originPath)
    # for file in files:
    #     if file.startswith("."):
    #         continue
    #     fdt = Dataset.readSimpleData(os.path.join(originPath, file), None)
    #     if fdt is None:
    #         XP.xprint_red(f"加载失败{file}")
    #     fDataList.append(fdt)
    dataset = DL.flowDataset(path=path, length=16384, step=8192, name="originWMS")
    # dataset.setDataset(fDataList, path=originPath)
    dataset.getDatasetInfo()
    dataloader = DL.flowDataLoader(dataset=dataset, batch_size=16, transform=train_trans)

    net = resnet1d34(class_num)
    net = net.to(device)
    weight = torch.load(modelPath, map_location=device)
    net.load_state_dict(weight)
    res = {}
    count = 0
    softmax = torch.nn.Softmax(1)
    while dataloader.isReadable():
        datas, _, paths = dataloader.getData()
        datas = datas.to(device)
        results = softmax(net(datas))
        for i in range(len(results)):
            if paths[i] not in res:
                res[paths[i]] = [0 for i in range(class_num)]
            for j in range(class_num):
                res[paths[i]][j] += results[i][j].item()
        count += 1
        if count % 100 == 0:
            dir = path.split("/")[-1]
            result_path = f"result{dir}.json"
            fp = open(result_path, "w")
            v = json.dumps(res)
            fp.write(v)
            fp.close()
            print(v)


def moveFile():
    labelRes = json.load(open("../v2data_process/WMS_Label.json"))
    fromPath = "/Users/lyn/codes/python/Flow_Identification/Dataset/new_data/origin/WMS_FILTER"
    toPath = "/Users/lyn/codes/python/Flow_Identification/Dataset/new_data/WMS"
    for file in labelRes:
        toDir = os.path.join(toPath, str(labelRes[file]["flowRegime"]))
        if not os.path.exists(toDir):
            os.makedirs(toDir, 0o777)
        shutil.copy(os.path.join(fromPath, file), toDir)


dpath = "/Users/lyn/codes/python/Flow_Identification/Dataset/Val/WMS_Simple"
paths = [
    os.path.join(dpath) for cls in os.listdir(dpath) if not dpath.startswith(".")
]
model = "/Users/lyn/codes/python/Flow_Identification/Flow_Identification/ex_result/train/20230704.160806_ResNet1d/weight.pth"
# do_process(dpath, model)
dealResultJson("/Users/lyn/codes/python/Flow_Identification/Flow_Identification/script/dataset/resultWMS_Simple.json")
# moveFile()
