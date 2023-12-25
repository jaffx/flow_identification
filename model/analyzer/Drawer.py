from matplotlib import pyplot as plt
import os


class Drawer:
    def __init__(self):
        self.pltInit()
        self.figWidth = 18
        self.figHeight = 12

    def pltShow(self, *args, **kwargs):
        self.pltFormat(*args, **kwargs)
        plt.show()

    def pltFormat(self, title="", footer=None, xLabel='X', yLabel='Y', legend=True):
        """
        绘图格式化
        @param title: str 标题
        @param footer: None|str 脚注
        @param xLabel: str X轴文字，系列名
        @param yLabel: str Y轴文字，系列名
        @param legend: bool 是否添加图例
        """
        # 设置标题
        plt.title(title, fontsize=40)
        # 设置图例
        # loc=0，自行选择位置
        if legend:
            plt.legend(framealpha=1, fontsize=30, loc=0)
        # 设置坐标轴
        plt.xlabel(xLabel, fontsize=30)
        plt.ylabel(yLabel, fontsize=30)
        plt.tick_params(axis='x', width=2, size=6)
        plt.tick_params(axis='y', width=2, size=6)
        plt.xticks(size=25)
        plt.yticks(size=25)

        # 添加脚注
        if footer is not None:
            bottom_space = 0.05 * len(footer.split("\n"))
            bottom_space = max(min(bottom_space, 0.3), 0.15)
            plt.subplots_adjust(bottom=bottom_space)
        plt.annotate(footer,
                     xy=(0, 0), xytext=(0, 10),
                     xycoords=('axes fraction', 'figure fraction'),
                     textcoords='offset points',
                     size=20, ha='left', va='bottom')

    @staticmethod
    def readDataFromFile(path, idx=0, vType=float, head=True, length=None):
        """
        从path指定的文件中读取第idx列数据数据
        如果idx==0，则读取全部数据
        :param path: 文件路径
        :param idx: 第几列，0表示全部
        :param vType: 数据的类型，使用强制转换，默认float类型
        :param head: 数据文件是否包括表头，True则去掉第一行
        :return:
        """
        assert os.path.isfile(path), f"{path}文件不存在"
        datas = []
        with open(path) as fp:
            if not length:
                lines = fp.readlines()
            else:
                lines = fp.readlines()
            if head:
                lines = lines[1:]
            for line in lines:
                line = line.strip("\n")
                items = line.split("\t")
                if idx != 0:
                    assert len(items) >= idx, f"数据字段数量错误"
                    data = vType(items[idx - 1])
                    datas.append(data)
                else:
                    datas.append(items)
        fp.close()
        return datas

    def initAxis(self, ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(4)
        ax.spines['bottom'].set_linewidth(4)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

    def getFig(self, width=None, height=None):
        if not width or not height:
            width = self.figWidth
            height = self.figHeight
        return plt.figure(figsize=(width, height))

    def getAxis(self, fig=None, nRows=1, nCols=1, index=1):
        if fig is None:
            fig = self.getFig()
        ax = fig.add_subplot(nRows, nCols, index)
        self.initAxis(ax)
        return ax

    def pltInit(self):
        plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
        plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
        plt.rcParams['font.sans-serif'] = ['Songti SC']
        plt.rcParams['axes.unicode_minus'] = False

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
