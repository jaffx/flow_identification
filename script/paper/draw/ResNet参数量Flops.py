import numpy as np
import matplotlib.pyplot as plt
from model.analyzer import Drawer


class Alyer(Drawer.Drawer):
    def draw(self):
        # 生成一些示例数据
        x = np.arange(10)
        y1 = np.random.randint(10, size=10)
        y2 = np.random.randint(10, size=10)

        # 创建一个图形和两个y轴
        fig = self.getFig()
        ax1 = self.getAxis(fig)
        ax2 = ax1.twinx()
        # 绘制折线图
        line1 = ax1.plot(x, y1, label='参数量', color='royalblue', marker='o', ls='-.')
        line2 = ax2.plot(x, y2, label='计算量', color='tomato', marker=None, ls='--')

        # 设置x轴和y轴的标签，指明坐标含义
        ax1.set_xlabel('模型层数', fontdict={'size': 30})
        ax1.set_ylabel('参数量', fontdict={'size': 30})
        ax2.set_ylabel('计算量', fontdict={'size': 30})
        plt.title("ResNet模型对比", fontdict={'size': 30})
        plt.legend(framealpha=1, fontsize=30, loc=0)
        plt.xticks(size=25, )  # 使用fontdict来设置xticks的字体大小
        plt.yticks(size=25, )  # 使用fontdict来设置yticks的字体大小
        plt.show()


aly = Alyer()
aly.draw()
