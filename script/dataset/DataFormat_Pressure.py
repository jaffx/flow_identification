"""
@date   2023年05月28日
@brief  压力数据格式处理,获取某一列压力数据
"""

import os
import re
import json

IDX = 4

def isfloat(s:str):
    try:
        f = float(s)
        return f
    except:
        return False

def initPath(path):
    if not os.path.exists(path):
        # print(f"目标文件夹不存在，创建文件夹{path}")
        os.makedirs(path)

origin_pressure_path = "/Users/lyn/codes/python/Flow_Identification/Dataset/v4/Pressure/v4_Pressure_Source_A"
out_pressure_path = f"/Users/lyn/codes/python/Flow_Identification/Dataset/v4/Pressure/v4_Pressure_IDX{IDX}_Simple_A"

initPath(out_pressure_path)

for cls in os.listdir(origin_pressure_path):
    if cls.startswith("."):
        continue
    origin_cls_path = os.path.join(origin_pressure_path, cls)
    out_cls_path = os.path.join(out_pressure_path, cls)
    initPath(out_cls_path)
    files = os.listdir(origin_cls_path)
    for file in files:
        if file.startswith("."):
            continue
        fileName = file.split(".")[0]
        filePath = os.path.join(origin_cls_path, file)
        outFileName = os.path.join(out_cls_path, file)
        print(f"正在处理文件：{file}")
        with open(filePath) as rfp, open(outFileName, "w+") as wfp:
            lineNum = 0
            content = rfp.readlines()
            for line in content:
                line = line.strip()
                items = line.split("\t")
                if len(items) != 5:
                    print(f"[{file}@LINE{lineNum}]数据字段数量错误" + json.dumps(items))
                    continue
                if not isfloat(items[IDX]):
                    continue
                target_line = f"{float(items[IDX]):.4f}\n"
                lineNum += 1
                wfp.write(target_line)
            rfp.close()
            wfp.close()
        print(f"文件处理完成：{filePath}, 行数：{lineNum}")

