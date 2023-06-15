from matplotlib import pyplot as plt
from analysis.analyzer import Analyzer, alyEnum
import os


class Analyzer_Epoch_Train(Analyzer):
    def __init__(self, res_path):
        super(Analyzer_Epoch_Train, self).__init__(res_path)

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

    def do_aly(self):
        val_acc = self.getEpochAcc("val")
        train_acc = self.getEpochAcc("train")
        val_loss = self.getEpochLoss("val")
        train_loss = self.getEpochLoss("train")
        best_val_acc = max(val_acc)
        best_val_acc_idx = val_acc.index(best_val_acc)

        ax = self.getAxis()

        ax.plot(val_acc, label="Validation Accurate", **alyEnum.Style_Plot_Red_H)
        ax.plot(train_acc, label="Train Accurate", **alyEnum.Style_Plot_Black_H)

        ax.plot(best_val_acc_idx, best_val_acc, **alyEnum.Style_Plot_Best_Acc)
        plt.text(best_val_acc_idx, best_val_acc -0.05, f"{best_val_acc * 100:.2f}%",
                 **alyEnum.Style_Text_Best_Acc)
        # y
        ax.set_ylim(-0.1, 1.1)
        plt.yticks(self.getRange(0, 1, 0.2))
        # x
        ax.set_xlim(-1, self.getInfo("Epoch_Num") + 1)
        plt.xticks(self.getRange(0, self.getInfo("Epoch_Num"), self.getInfo("Epoch_Num")//10))
        plt.text(0, 0, "xyq")
        self.pltShow(title="Accurate", xlabel='Epoch', ylabel='Accurate')

        ax = self.getAxis()
        ax.plot(val_loss, label="Validation Loss", **alyEnum.Style_Plot_Red_H)
        ax.plot(train_loss, label="Train Loss", **alyEnum.Style_Plot_Black_H)
        ax.set_ylim(-0.05, 1.1)
        ax.set_xlim(-1, self.getInfo("Epoch_Num") + 1)
        plt.xticks(self.getRange(0, self.getInfo("Epoch_Num"), self.getInfo("Epoch_Num")//10))
        plt.yticks(self.getRange(0, 1, 0.2))
        self.pltShow(title="Loss", xlabel='Epoch', ylabel='Loss')
