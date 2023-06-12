"""
@date   2023年05月28日
@brief  压力数据格式处理
@desc
    从origin文件夹中读取数据，按行处理，将数据放到指定的文件夹中
    运行结果放到 script/v2data_process/process.log
"""

import os
import re
import json

def isfloat(s:str):
    try:
        f = float(s)
        return f
    except:
        return False

origin_pressure_path = "/Users/lyn/codes/python/Flow_Identification/Dataset/v2/origin/pressure"
out_pressure_path = "/Users/lyn/codes/python/Flow_Identification/Dataset/v2/Pressure/Pressure_IDX3"
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
            content = rfp.readlines()
            for line in content:

                line = line.strip()
                # print(line)
                items = line.split("\t")
                # print(items)
                if len(items) != 5:
                    print(f"[{file}@LINE{lineNum}]数据字段数量错误" + json.dumps(items))
                    continue
                if not isfloat(items[3]):
                    continue
                target_line = f"{float(items[3]):.4f}\n"
                lineNum += 1
                wfp.write(target_line)
        except Exception as e:
            process_desc["err"].append({"file": file, "err": str(e), "lineNum": lineNum, "content": line})
            print(e)
        rfp.close()
        wfp.close()
    print(f"文件处理完成：{filePath}, 行数：{lineNum}")
    process_desc["succ"].append(file)

with open("script/v2data_process/process.log", "w+") as fp:
    json.dump(process_desc, fp)
