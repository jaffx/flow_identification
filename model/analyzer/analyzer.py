import os

import yaml
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

from . import alyEnum
import matplotlib.font_manager


class Analyzer:
    """
    Analyzer 分析器
    提供数据筛选、数据分析、图表绘制等能力，对训练结果进行格式化分析
    """
    def __init__(self, result_path):
        """
        指定一个result文件路径，该路径下默认存在info.yaml文件
        :param result_path:
        """
        self.info = None
        self.pltInit()
        self.result_path = result_path

    @staticmethod
    def readDataFromFile(path, idx=0, vtype=float, head=True):
        """
        从path指定的文件中读取第idx列数据数据
        如果idx==0，则读取全部数据
        :param path: 文件路径
        :param idx: 第几列，0表示全部
        :param vtype: 数据的类型，使用强制转换，默认float类型
        :param head: 数据文件是否包括表头，True则去掉第一行
        :return:
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
        """
        读取基本信息，默认读取info.yaml的内容
        :return:
        """
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
        """
        检查数据格式是否完整
        :return:
        """
        files = os.listdir(self.result_path)
        # 检查数据文件存在
        if "train_iter" not in files or "val_iter" not in files or "info.yaml" not in files or "epoch" not in files:
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

        epoch_num = self.getInfo("Epoch_Num")
        if epoch_num is None:
            epoch_num = 50
        epoch_num = min(epoch_num, 50)
        with open(os.path.join(self.result_path, "epoch")) as fp:
            content = fp.readlines()
            if len(content) < epoch_num + 1:
                return False
        return True

    @staticmethod
    def getAxis():
        fig = plt.figure(figsize=(18, 12))
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

    def getDeaultFooter(self):
        lines = []
        lines.append(
            f"Dataset={self.getInfo('Dataset')} Epoch={self.getInfo('Epoch_Num')} "
            f"BatchSize={self.getInfo('Batch_Size')} Length={self.getInfo('Data_Length')} "
            f"Step={self.getInfo('Sampling_Step')} LR={self.getInfo('Learn_Rate')}"
        )

        # mes_son = [mes_str[i: i + 8] for i in range(0, len(mes_str), 8)]
        lines.append(f"Ttrans={self.getInfo('Train_Transform')}")
        lines.append(f"Vtrans{self.getInfo('Val_Transform')}")

        content = "    ".join(lines);
        line_length = 120
        content = [content[i: i + line_length] for i in range(0, len(content), line_length)]

        return "\n".join(content)

    def pltShow(self, title, footer=None, xlabel='X', ylabel='Y', save=False):
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

        if footer is True:
            footer = self.getDeaultFooter()
            bottom_space = 0.05 * len(footer.split("\n"))
            bottom_space = max(min(bottom_space, 0.3), 0.15)
            plt.subplots_adjust(bottom=bottom_space)
        plt.annotate(footer,
                     xy=(0, 0), xytext=(0, 10),
                     xycoords=('axes fraction', 'figure fraction'),
                     textcoords='offset points',
                     size=20, ha='left', va='bottom')
        plt.show()

    def do_aly(self):
        pass

    @staticmethod
    def getRange(start, end, step):
        ranges = []
        i = start
        while i <= end:
            if i <= end:
                ranges.append(i)
            else:
                return ranges
            i += step
        return ranges
