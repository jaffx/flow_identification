"""
@date   2023年05月28日
@brief  压力数据格式处理
@desc
    从origin文件夹中读取数据，按行处理，将数据放到指定的文件夹中
    运行结果放到 script/new_data_process/process.log
"""

import os
import re
import json

origin_pressure_path = "/Users/lyn/codes/python/Flow_Identification/Dataset/new_data/origin/pressure"
out_pressure_path = "/Users/lyn/codes/python/Flow_Identification/Dataset/new_data/DiffPressure_IDX3-4"
files = os.listdir(origin_pressure_path)
if not os.path.exists(out_pressure_path):
    print(f"目标文件夹不存在，创建文件夹{out_pressure_path}")
    os.makedirs(out_pressure_path)
process_desc = {
    "succ": [],
    "name_err": [],
    "err": []
}

for file in files:
    if re.match("G\d+L\d+\.txt", file) is None:
        process_desc["name_err"].append(file)
        print(f"{file} 名称不匹配")
        continue
    fileName = file.split(".")[0]
    filePath = os.path.join(origin_pressure_path, file)
    outFileName = os.path.join(out_pressure_path, file)
    print(f"正在处理文件：{file}")
    with open(filePath) as rfp, open(outFileName, "w+") as wfp:
        lineNum = 0
        try:
            while rfp:
                lineNum += 1
                line = rfp.readline()
                if line == "":
                    break
                line = line.strip()
                # print(line)
                items = line.split("\t")
                # print(items)
                if len(items) != 5:
                    print(f"[{file}@LINE{lineNum}]数据字段数量错误" + json.dumps(items))
                if not items[3].isnumeric() or not items[4].isnumeric():
                    continue
                target_line = f"{float(items[3])-float(items[4]):.4f}\n"
                wfp.write(target_line)
        except Exception as e:
            process_desc["err"].append({"file": file, "err": str(e), "lineNum": lineNum, "content": line})
        rfp.close()
        wfp.close()
    print(f"文件处理完成：{filePath} -> {outFileName}")
    process_desc["succ"].append(file)

with open("script/new_data_process/process.log", "w+") as fp:
    json.dump(process_desc, fp)
