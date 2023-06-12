import os
from matplotlib import pyplot as plt


class Analyzer:
    def __init__(self):
        pass

    def readDataFromFile(self, path, idx):
        assert os.path.isfile(path), f"{path}文件不存在"
        datas = []
        with open(path) as fp:
            lines = fp.readlines()
            for line in lines:
                line = line.strip("\n")
                items = line.split("\t")
                assert len(items) >= idx, f"数据字段数量错误"
                data = float(items[idx])
                datas.append(datas)
            fp.close()
        return datas

    def