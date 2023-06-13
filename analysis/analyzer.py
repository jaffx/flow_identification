import os

import yaml
from matplotlib import pyplot as plt
from analysis import alyEnum


class Analyzer:
    def __init__(self, result_path):
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

    def getEpochAcc(self, type="val"):
        epoch_path = os.path.join(self.result_path, "epoch")
        assert type in ("val", "train")
        assert os.path.isfile(epoch_path), f'epoch文件不存在{epoch_path}'
        idx = alyEnum.Epoch_IDX_ValAcc if type == "val" else alyEnum.Epoch_IDX_TrainAcc
        acc = self.readDataFromFile(epoch_path, idx, float)
        return acc

    def getEpochLoss(self, type="val"):
        epoch_path = os.path.join(self.result_path, "epoch")
        assert type in ("val", "train")
        assert os.path.isfile(epoch_path), f'epoch文件不存在{epoch_path}'
        idx = alyEnum.Epoch_IDX_ValLoss if type == "val" else alyEnum.Epoch_IDX_TrainLoss
        loss = self.readDataFromFile(epoch_path, idx, float)
        return loss

    def alyEpoch(self):
        val_acc = self.getEpochAcc("val")
        train_acc = self.getEpochAcc("train")
        val_loss = self.getEpochLoss("val")
        train_loss = self.getEpochLoss("train")
        plt.title("Loss")
        plt.plot(val_loss)
        plt.plot(train_loss)
        plt.show()

        plt.title("Accurate")
        plt.plot(val_acc)
        plt.plot(train_acc)
        plt.show()


path = "/Users/lyn/codes/python/Flow_Identification/Flow_Identification/bk_result/train/2023-06-12 02.01.39 [ResNet1d]"
aly = Analyzer(path)
aly.alyEpoch()
