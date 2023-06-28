"""
@brief 将wms的.epst从Source格式转化为Simple格式
"""
import os

originPath = "/Users/lyn/codes/python/Flow_Identification/Dataset/v4/WMS/v4_WMS_Label_Source_A"
outPath = "/Users/lyn/codes/python/Flow_Identification/Dataset/v4/WMS/v4_WMS_Label_Simple_A"


def checkAndInitPath(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"创建文件夹{path}")

for cls in os.listdir(originPath):
    if cls.startswith("."):
        continue
    cls_path = os.path.join(originPath, cls)
    out_cls_path = os.path.join(outPath, cls)
    checkAndInitPath(out_cls_path)
    print(f"处理分类{cls}")
    for file in os.listdir(cls_path):
        if file.startswith("."):
            continue
        print(f"处理文件{file}")
        file_path = os.path.join(cls_path, file)
        out_file_path = os.path.join(out_cls_path, file)
        with open(file_path) as fp, open(out_file_path, "w") as wfp:
            content = fp.readlines()[2:]
            for line in content:
                items = line.split(" ")
                value = float(items[-1])
                wfp.write(f"{value}\n")
        fp.close()
        wfp.close()
