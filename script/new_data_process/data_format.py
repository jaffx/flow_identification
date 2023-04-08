# 处理数据的格式
import os
import re
import shutil

dataset_dir = "/Users/lyn/codes/python/Flow_Identification/Dataset/new_data"
target_dir = "/Users/lyn/codes/python/Flow_Identification/Dataset/new_data/pressure_wms"
pressure_dir = os.path.join(dataset_dir, "pressure")
wms_dir = os.path.join(dataset_dir, "WMS")