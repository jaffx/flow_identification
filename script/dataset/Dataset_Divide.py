"""
@brief 数据集分割脚本，将A类型数据集转化为B类型数据集
@date 2023年06月07日
@author xuyongqi
"""

import os
import sys


def checkAndInitPath(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"创建文件夹:{path}")


datasetPath = "/Users/lyn/codes/python/Flow_Identification/Dataset/v4/Pressure/v4_DiffPressure_IDX3-4_Simple_A"
targetPath = "/Users/lyn/codes/python/Flow_Identification/Dataset/v4/Pressure/v4_DiffPressure_IDX3-4_Simple_B"

args = sys.argv
if len(args) == 2:
    datasetPath = args[0]
    targetPath = args[1]

if not os.path.isdir(datasetPath):
    print("数据集路径不存在")
    exit(1)

# 训练集占比
trainSetRate = 0.7
trainPath = os.path.join(targetPath, "train")
valPath = os.path.join(targetPath, "val")

checkAndInitPath(trainPath)
checkAndInitPath(valPath)

for cls in os.listdir(datasetPath):
    # 初始化子类文件夹
    if cls.startswith("."):
        continue
    print(f"处理子类{cls}")
    cls_path = os.path.join(datasetPath, cls)
    train_cls_path = os.path.join(trainPath, cls)
    val_cls_path = os.path.join(valPath, cls)
    checkAndInitPath(train_cls_path)
    checkAndInitPath(val_cls_path)
    for file in os.listdir(cls_path):
        if file.startswith("."):
            continue
        print(f"处理文件{file}")
        file_path = os.path.join(cls_path, file)
        train_file = os.path.join(train_cls_path, file)
        val_file = os.path.join(val_cls_path, file)
        with open(file_path) as rfp, open(train_file, "w") as tfp, open(val_file, "w") as vfp:
            content = rfp.readlines()[2:]
            dataNums = len(content)
            splitNum = int(dataNums * trainSetRate)
            trainSet = content[:splitNum]
            valSet = content[splitNum:]
            for line in trainSet:
                tfp.write(line)
            for line in valSet:
                vfp.write(line)
            tfp.close()
            vfp.close()
