"""
对放置结果的目录进行筛选，并将完整的实验结果移动到目标文件夹
"""
import os
import shutil
import sys

sys.path.append(".")
from analysis.analyzer import Analyzer

check_result_path = "bk_result/train"
target_result_path = "ex_result/train"


def initPath(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"创建文件夹{path}")


initPath(target_result_path)
for res in os.listdir(check_result_path):
    res_path = os.path.join(check_result_path, res)
    if res.startswith(".") or res.startswith("_"):
        continue
    aly = Analyzer(res_path)
    if not aly.checkResult():
        print(f"文件不完整,跳过{res}")
        continue
    target_path = os.path.join(target_result_path, res)
    if os.path.exists(target_path):
        print(f"文件已存在,跳过{res}")
        continue
    shutil.copytree(res_path, target_path)
    print(f"处理完成{res}")
