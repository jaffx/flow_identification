import json
import os

import yaml


class config:
    def __init__(self, path):
        self.path = path
        self.conf = self.getConf(path)

    @staticmethod
    def getConf(path: str):
        assert os.path.isfile(path), f"conf path {path} is not file!"
        fp = open(path, 'r')
        if path.endswith(".json"):
            conf = json.load(fp)
        elif path.endswith(".yaml") or path.endswith(".yml"):
            conf = yaml.load(fp, yaml.SafeLoader)
        else:
            raise Exception("文件必须是json或者是yaml类型")
        return conf

    def get(self, conf_path: str = "/"):
        """
        获取配置信息
        """
        assert conf_path, "配置路径不能为空"
        conf_paths = conf_path.split("/")
        config = self.conf
        for conf_key in conf_paths:
            if conf_key == "":
                continue
            if conf_key in config:
                config = config[conf_key]
            else:
                return None
        return config


