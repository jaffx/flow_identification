"""
流型持液率样本曲线
"""

from matplotlib import pyplot as plt
from model.analyzer.Analyzer import Drawer
import numpy as np


class ppAly(Drawer.Drawer):
    dataPaths = [
        # "/Users/lyn/codes/python/Flow_Identification/Dataset/v1/WMS_Simple_B/train/0/L50G0_Y_Sensor_2.epst",
        # "/Users/lyn/codes/python/Flow_Identification/Dataset/v1/WMS_Simple_B/train/1/L50G230_Y_Sensor_2.epst",
        "/Users/lyn/codes/python/Flow_Identification/Dataset/v1/WMS_Simple_B/train/2/L50G300_Y_Sensor_2.epst",
        # "/Users/lyn/codes/python/Flow_Identification/Dataset/v1/WMS_Simple_B/train/3/L50G400_Y_Sensor_2.epst",
    ]
    dataLength = 12000
    timeInterval = 0.008

    def __init__(self):
        super(ppAly, self).__init__()
        self.samples = None
        self.xRange = self.getRange(0, (self.dataLength + 10) * self.timeInterval, self.timeInterval)[:self.dataLength]
        self.fig = self.getFig(width=30, height=5)

    def getSamples(self):
        if self.samples is None:
            self.samples = [np.array(self.readSample(p)) for p in self.dataPaths]
        return self.samples

    def readSample(self, path):
        with open(path) as fp:
            data = []
            for i in range(self.dataLength):
                data.append(100 - float(fp.readline()))
        return data

    def getAx(self, row, index):
        return self.getAxis(self.fig, row, 1, index)

    def plotOrigin(self):
        ax = self.getAx(1, 1)
        data = self.getSamples()[0]
        ax.plot(self.xRange, data, color="black", linewidth=2)
        plt.gca().spines['right'].set_color('none')  # 隐藏右侧坐标轴

        plt.gca().spines['top'].set_color('none')  # 隐藏上方坐标轴
        plt.gca().spines['left'].set_color('none')  # 隐藏左侧坐标轴
        plt.gca().spines['bottom'].set_color('none')  # 隐藏下方坐标轴

        plt.xticks(())  # 隐藏x轴刻度标签
        plt.yticks(())  # 隐藏y轴刻度标签

    def showNormalization(self):
        self.plotOrigin()
        plt.show()
        # plt.savefig("/Users/lyn/Desktop/myplot.png")


aly = ppAly()
aly.showNormalization()
