import os.path

import yaml


class train_setting:
    def __init__(self, file_path="settings/base.yaml"):

        self.file_path = file_path

        self.learn_rate = None
        self.sampling_step = None
        self.dataset = None
        self.data_length = None
        self.epoch_num = None
        self.batch_size = None
        self.device_name = None

        self.load()

    def Init(self):
        self.learn_rate = None
        self.sampling_step = None
        self.dataset = None
        self.data_length = None
        self.epoch_num = None
        self.batch_size = None
        self.device_name = None

    def load(self):
        assert os.path.exists(self.file_path), f"setting链接文件[{self.file_path}]不存在"
        with open(self.file_path) as fp:
            _set: dict = yaml.full_load(fp)
        for key in _set.keys():
            if key == "dataset":
                self.dataset = _set[key]
            elif key == "data_length":
                self.data_length = _set[key]
            elif key == "sampling_step":
                self.sampling_step = _set[key]
            elif key == "epoch_num":
                self.epoch_num = _set[key]
            elif key == "batch_size":
                self.batch_size = _set[key]
            elif key == "learn_rate":
                self.learn_rate = _set[key]
            elif key == "device_name":
                self.device_name = _set[key]
            else:
                continue

    def __str__(self):
        return str(self.to_dict())

    def check(self):
        pass

    def to_dict(self):
        return {
            "file_path": self.file_path,
            "dataset": self.dataset,
            "data_length": self.data_length,
            "sampling_step": self.sampling_step,
            "epoch_num": self.epoch_num,
            "learn_rate": self.learn_rate,
            "batch_size": self.batch_size,
            "device_name": self.device_name
        }
