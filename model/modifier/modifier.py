from . import *


class Modifier:
    @staticmethod
    def checkConf(conf: dict):
        """
        :param conf 配置
        """
        if "desc" not in conf or "mod" not in conf:
            return False
        return True

    def __init__(self):
        pass
