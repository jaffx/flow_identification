import os

import yaml
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

from analysis import alyEnum
import matplotlib.font_manager


class Analyzer:

    def __init__(self, result_path):
        self.info = None
        self.pltInit()
        self.result_path = result_path

    @staticmethod
    def readDataFromFile(path, idx=0, vtype=float, head=True):
        """
        :param path
        :param idx
        :param vtype
        :param head
        """
        assert os.path.isfile(path), f"{path}文件不存在"
        datas = []
        with open(path) as fp:
            lines = fp.readlines()
            if head:
                lines = lines[1:]
            for line in lines:
                line = line.strip("\n")
                items = line.split("\t")
                if idx != 0:
                    assert len(items) >= idx, f"数据字段数量错误"
                    data = vtype(items[idx - 1])
                    datas.append(data)
                else:
                    datas.append(items)
        fp.close()
        return datas

    def loadInfo(self):
        info_path = os.path.join(self.result_path, "info.yaml")
        assert os.path.isfile(info_path), f"info文件不存在{info_path}"
        with open(info_path) as fp:
            info = yaml.safe_load(fp)
            fp.close()
        if "Epoch_Num" not in info:
            info["Epoch_Num"] = 50
        return info

    def getInfo(self, key):
        if not self.info:
            self.info = self.loadInfo()
        if key in self.info:
            return self.info[key]
        return None

    def checkResult(self) -> bool:
        files = os.listdir(self.result_path)
        # 检查数据文件存在
        if "train_iter" not in files or "val_iter" not in files or "info" not in files or "epoch" not in files:
            return False
        # 检查权重文件存在
        find_pth = False
        for file in files:
            if file.endswith(".pth"):
                find_pth = True
                break
        if not find_pth:
            return False
        # 检查训练是否完成
        info = self.loadInfo()
        epoch_num = info["Epoch_Num"]
        with open(os.path.join(self.result_path)) as fp:
            content = fp.readlines()
            if len(content) != epoch_num + 1:
                return False
        return True

    @staticmethod
    def getAxis():
        fig = plt.figure("title", figsize=(18, 12))
        ax = fig.add_subplot(111)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(4)
        ax.spines['bottom'].set_linewidth(4)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        return ax

    @staticmethod
    def pltInit():
        plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
        plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
        plt.rcParams['font.sans-serif'] = ['Songti SC']
        plt.rcParams['axes.unicode_minus'] = False

    @staticmethod
    def pltShow(title, xlabel='X', ylabel='Y'):
        # 设置标题
        plt.title(title, fontsize=40)
        # 设置图例
        # loc 0 best
        plt.legend(framealpha=1, fontsize=30, loc=0)
        # 设置坐标轴
        plt.xlabel(xlabel, fontsize=30)
        plt.ylabel(ylabel, fontsize=30)
        plt.tick_params(axis='x', width=2, size=6)
        plt.tick_params(axis='y', width=2, size=6)
        plt.xticks(size=25)
        plt.yticks(size=25)
        plt.show()

    def do_aly(self):
        pass

    @staticmethod
    def getRange(start, end, step):
        limit = 1000
        ranges = []
        i = start
        while i <= end:
            if limit <= 0:
                return ranges
            limit -= 1
            if i <= end:
                ranges.append(i)
            else:
                return ranges
            i += step
        return ranges
