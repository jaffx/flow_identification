"""
@date 2023年06月04日
@author xuyongqi
加载一个网络模型，读取WMS数据，进行识别并将识别结果保存到result.json文件中
"""

import json
import os
import shutil
import sys

sys.path.append(".")
import torch
from tools.transforms import Preprocess as PP
from tools.transforms import BaseTrans as BT
from tools.DataLoader import DataLoader as DL
from tools.Dataset import Dataset
from tools.xyq import x_printer as XP
from model.Res1D import resnet1d34

sys.path.append(".")
from model.Res1D import resnet1d34


def dealResultJson():
    result = json.load(open("result.json"))
    ret = {}
    for file in result:
        props = result[file]
        sumProp = sum(props)
        maxProp = max(props)
        maxPropIdx = 0
        for i in range(len(props)):
            if props[i] == maxProp:
                maxPropIdx = i
        fileName = file.split("/")[-1]
        ret[fileName] = {
            "prop": f"{int(maxProp / sumProp * 100)}%",
            "flowRegime": maxPropIdx
        }
    fp = open("WMS_Label.json", "w+")
    json.dump(ret, fp)
    fp.close()


def do_process():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    originPath = "/Users/lyn/codes/python/Flow_Identification/Dataset/new_data/origin/WMS_FILTER"
    modelPath = "/Users/lyn/codes/python/Flow_Identification/Flow_Identification/result/2023-05-29 12.10.18 [ResNet1d]/2023-05-29 12.10.18 [ResNet1d].pth"
    train_trans = BT.transfrom_set([
        PP.normalization(),
        BT.toTensor()
    ])

    fDataList = []
    files = os.listdir(originPath)
    for file in files:
        fdt = Dataset.readWMSFile(os.path.join(originPath, file), None)
        if fdt is None:
            XP.xprint_red(f"加载失败{file}")
        fDataList.append(fdt)
    dataset = DL.flowDataset(path=None, length=16384, step=8192, name="originWMS")
    dataset.setDataset(fDataList, path=originPath)
    dataset.getDatasetInfo()
    dataloader = DL.flowDataLoader(dataset=dataset, batch_size=16, transform=train_trans)

    net = resnet1d34(4)
    net = net.to(device)
    weight = torch.load(modelPath, map_location=device)
    net.load_state_dict(weight)
    res = {}
    count = 0
    softmax = torch.nn.Softmax(1)
    while dataloader.getReadable():
        datas, _, paths = dataloader.getData()
        datas = datas.to(device)
        results = softmax(net(datas))
        for i in range(len(results)):
            if paths[i] not in res:
                res[paths[i]] = [0, 0, 0, 0]
            for j in range(4):
                res[paths[i]][j] += results[i][j].item()
        count += 1
        if count % 100 == 0:
            fp = open("result.json", "w+")
            v = json.dumps(res)
            fp.write(v)
            fp.close()
            print(v)


def moveFile():
    labelRes = json.load(open("WMS_Label.json"))
    fromPath = "/Users/lyn/codes/python/Flow_Identification/Dataset/new_data/origin/WMS_FILTER"
    toPath = "/Users/lyn/codes/python/Flow_Identification/Dataset/new_data/WMS"
    for file in labelRes:
        toDir = os.path.join(toPath, str(labelRes[file]["flowRegime"]))
        if not os.path.exists(toDir):
            os.makedirs(toDir,0o777)
        shutil.copy(os.path.join(fromPath, file), toDir)


# do_process()
# dealResultJson()
moveFile()