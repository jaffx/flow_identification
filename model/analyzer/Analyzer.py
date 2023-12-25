import os
import yaml
from . import *


class Analyzer(Drawer.Drawer):
    """
    Analyzer 分析器
    提供数据筛选、数据分析、图表绘制等能力，对训练结果进行格式化分析
    """

    def __init__(self, path):
        """
        指定一个result文件路径，该路径下默认存在info.yaml文件
        :param path:
        """
        super(Analyzer, self).__init__()
        self.info = None
        self.pltInit()
        self.path = path

    def getInfo(self, key):
        if not self.info:
            self.info = self.loadInfo()
        if key in self.info:
            return self.info[key]
        return None

    def checkResult(self) -> bool:
        if not os.path.isdir(self.path):
            return False
        if not os.path.isfile(os.path.join(self.path, "info.yaml")):
            return False
        return True

    def do_aly(self):
        pass

    def loadInfo(self):
        """
        读取基本信息，默认读取info.yaml的内容
        :return:
        """
        info_path = os.path.join(self.path, "info.yaml")
        assert os.path.isfile(info_path), f"info文件不存在{info_path}"
        with open(info_path) as fp:
            info = yaml.safe_load(fp)
            fp.close()
        if "Epoch_Num" not in info:
            info["Epoch_Num"] = 50
        return info
