"""
画图展示正则化处理对数据样本的影响
"""

from matplotlib import pyplot as plt
from model.analyzer.analyzer import Drawer
import numpy as np


class ppAly(Drawer):
    dataPaths = [
        # "/Users/lyn/codes/python/Flow_Identification/Dataset/v1/WMS_Simple_B/train/0/L50G0_Y_Sensor_2.epst",
        "/Users/lyn/codes/python/Flow_Identification/Dataset/v1/WMS_Simple_B/train/1/L50G230_Y_Sensor_2.epst",
        "/Users/lyn/codes/python/Flow_Identification/Dataset/v1/WMS_Simple_B/train/2/L50G300_Y_Sensor_2.epst",
        # "/Users/lyn/codes/python/Flow_Identification/Dataset/v1/WMS_Simple_B/train/3/L50G400_Y_Sensor_2.epst",
    ]
    dataLength = 12000
    timeInterval = 0.008

    def __init__(self):
        super(ppAly, self).__init__()
        self.samples = None
        self.xRange = self.getRange(0, (self.dataLength + 10) * self.timeInterval, self.timeInterval)[:self.dataLength]
        self.fig = self.getFig(width=30, height=20)

    def readSample(self, path):
        with open(path) as fp:
            data = []
            for i in range(self.dataLength):
                data.append(100 - float(fp.readline()))
        return data

    def getAx(self, row, index):
        return self.getAxis(self.fig, row, 1, index)

    def getSamples(self):
        if self.samples is None:
            self.samples = [np.array(self.readSample(p)) for p in self.dataPaths]
        return self.samples

    def plotOrigin(self):
        ax = self.getAx(2, 1)
        data = self.getSamples()
        ax.plot(self.xRange, data[0], label=f"样本1", color="black", linewidth=2)
        ax.plot(self.xRange, data[1], label=f"样本2", color="red", linewidth=2)
        self.pltFormat(title="原始数据样本", xLabel="时间", yLabel="持液率")

    def plotNormal(self):
        ax = self.getAx(2, 2)
        data = self.getSamples()
        mean = np.mean(data[0])
        std = np.std(data[0])
        ax.plot(self.xRange, (data[0] - mean) / std, label=f"样本1", color="black", linewidth=2)
        mean = np.mean(data[1])
        std = np.std(data[1])
        ax.plot(self.xRange, (data[1] - mean) / std, label=f"样本2", color="red", linewidth=2)

        self.pltFormat(title="正则化处理后的样本", xLabel="时间", yLabel="正则化持液率", )

    def showNormalization(self):
        self.plotOrigin()
        self.plotNormal()
        plt.savefig("/Users/lyn/念书/大论文/画图/正则化作用示意图.png")
        # plt.show()


aly = ppAly()
aly.showNormalization()
