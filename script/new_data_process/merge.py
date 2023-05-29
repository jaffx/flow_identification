"""
@date 2023年03月28日
@brief 压力信号和wms数据处理
将同种工况的wms数据和压力数据归并到一起

"""
import os
import re
import shutil

dataset_dir = "/Users/lyn/codes/python/Flow_Identification/Dataset/new_data"
target_dir = "/Users/lyn/codes/python/Flow_Identification/Dataset/new_data/pressure_wms"
pressure_dir = os.path.join(dataset_dir, "pressure")
wms_dir = os.path.join(dataset_dir, "WMS")


def process_pressure_datas():
    # 处理压力数据
    files = os.listdir(pressure_dir)
    pattern = re.compile("G(?P<Gas>\d+)L(?P<Liquid>\d+).txt")
    for file in files:
        file.split(".")
        match_ret = pattern.match(file)
        if (match_ret):
            Gas = int(match_ret.group("Gas"))
            Liquid = int(match_ret.group("Liquid"))
            target_file_path = os.path.join(target_dir, f"G{Gas}L{Liquid}")
            if not os.path.exists(target_file_path):
                os.makedirs(target_file_path)
            shutil.copy(os.path.join(pressure_dir, file), os.path.join(target_file_path, "pressure.txt"))


process_pressure_datas()
