from matplotlib import pyplot as plt
from analysis.analyzer import Analyzer, alyEnum
import os


class Analyzer_Iter_Train(Analyzer):
    def __init__(self, res_path):
        super(Analyzer_Iter_Train, self).__init__(res_path)

    def getAcc(self, type="train"):
        assert type in ("train", "val"), "type must be train or val"
        if type == "train":
            iter_path = os.path.join(self.result_path, "train_iter")
        else:
            iter_path = os.path.join(self.result_path, "val_iter")
        accs = self.readDataFromFile(iter_path, alyEnum.Iter_IDX_Acc)
        return accs

    def getAvgLoss(self, type="train"):
        assert type in ("train", "val"), "type must be train or val"
        if type == "train":
            iter_path = os.path.join(self.result_path, "train_iter")
        else:
            iter_path = os.path.join(self.result_path, "val_iter")
        avg_loss = self.readDataFromFile(iter_path, alyEnum.Iter_IDX_Loss)
        return avg_loss

    def do_aly(self, save=False):
        train_acc, val_acc = self.getAcc("train"), self.getAcc("val")
        train_loss, val_loss = self.getAvgLoss("train"), self.getAvgLoss("val")
        train_x = self.getRange(1, len(train_acc), 1)
        val_x = self.getRange(1, len(train_acc) + 1, len(train_acc) / len(val_acc))[:len(val_acc)]

        ax = self.getAxis()
        ax.scatter(train_x, train_loss, s=10, color="black", marker=".", label="Train Loss", )
        ax.scatter(val_x, val_loss, s=10, color="red", marker=".", label="Val Loss", )
        ax.set_ylim(-0.05, 1.1)
        self.pltShow(title="Iteration Loss", save=f"figs/iter_loss_{self.result_path.split('/')[-1]}.png")

        ax = self.getAxis()
        ax.scatter(train_x, train_acc, s=10, color="black", marker=".", label="Train Accurate", )
        ax.scatter(val_x, val_acc, s=10, color="red", marker=".", label="Val Accurate", )
        ax.set_ylim(-0.05, 1.1)
        # self.pltShow(title="Iteration Accurate", save=f"figs/iter_acc_{self.result_path.split('/')[-1]}.png")
        self.pltShow(title="Iteration Accurate", save=save)
