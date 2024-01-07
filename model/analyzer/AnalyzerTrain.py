from . import *


class AnalyzerTrain(Analyzer.Analyzer):
    def __init__(self, path):
        super(AnalyzerTrain, self).__init__(path)

    def getDefaultFooter(self):
        lines = [f"Dataset={self.getInfo('Dataset')} Epoch={self.getInfo('Epoch_Num')} "
                 f"BatchSize={self.getInfo('Batch_Size')} Length={self.getInfo('Data_Length')} "
                 f"Step={self.getInfo('Sampling_Step')} LR={self.getInfo('Learn_Rate')}",
                 f"Ttrans={self.getInfo('Train_Transform')}", f"Vtrans{self.getInfo('Val_Transform')}"]

        # mes_son = [mes_str[i: i + 8] for i in range(0, len(mes_str), 8)]

        content = "    ".join(lines);
        line_length = 120
        content = [content[i: i + line_length] for i in range(0, len(content), line_length)]

        return "\n".join(content)

    def checkResult(self) -> bool:
        """
        检查数据格式是否完整
        :return:
        """
        if not super(AnalyzerTrain, self).checkResult():
            return False
        files = os.listdir(self.path)
        # 检查数据文件存在
        if "train_iter" not in files \
                or "val_iter" not in files \
                or "epoch" not in files:
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
        epoch_num = self.getInfo("Epoch_Num")
        if epoch_num is None:
            epoch_num = 50
        epoch_num = min(epoch_num, 50)
        with open(os.path.join(self.path, "epoch")) as fp:
            content = fp.readlines()
            if len(content) < epoch_num + 1:
                return False
        return True


class AnalyzerTrainEpoch(AnalyzerTrain):
    def __init__(self, path):
        super(AnalyzerTrainEpoch, self).__init__(path)
        self.epochPath = os.path.join(self.path, "epoch")
        assert os.path.isfile(self.epochPath), f'epoch文件不存在{self.epochPath}'

    def getEpochAcc(self, method="val"):
        assert method in ("val", "train")
        idx = DrawEnum.Epoch_IDX_ValAcc if method == "val" else DrawEnum.Epoch_IDX_TrainAcc
        acc = self.readDataFromFile(self.epochPath, idx, float)
        return acc

    def getEpochLoss(self, method="val"):
        assert method in ("val", "train")
        idx = DrawEnum.Epoch_IDX_ValLoss if method == "val" else DrawEnum.Epoch_IDX_TrainLoss
        loss = self.readDataFromFile(self.epochPath, idx, float)
        return loss

    def getBestAcc(self, method="val"):
        return max(self.getEpochAcc(method))



    def do_aly(self):
        val_acc = self.getEpochAcc("val")
        train_acc = self.getEpochAcc("train")
        val_loss = self.getEpochLoss("val")
        train_loss = self.getEpochLoss("train")
        best_val_acc = max(val_acc)
        best_val_acc_idx = val_acc.index(best_val_acc)

        ax = self.getAxis()

        ax.plot(val_acc, label="Validation Accurate", **DrawEnum.Style_Plot_Red_H)
        ax.plot(train_acc, label="Train Accurate", **DrawEnum.Style_Plot_Black_H)

        ax.plot(best_val_acc_idx, best_val_acc, **DrawEnum.Style_Plot_Best_Acc)
        plt.text(best_val_acc_idx, best_val_acc - 0.05, f"{best_val_acc * 100:.2f}%",
                 **DrawEnum.Style_Text_Best_Acc)
        # y
        ax.set_ylim(-0.1, 1.1)
        plt.yticks(self.getRange(0, 1, 0.2))
        # x
        ax.set_xlim(-1, self.getInfo("Epoch_Num") + 1)
        plt.xticks(self.getRange(0, self.getInfo("Epoch_Num"), self.getInfo("Epoch_Num") // 10))
        plt.text(0, 0, "xyq")
        self.pltShow(title="Accurate", xLabel='Epoch', yLabel='Accurate')

        ax = self.getAxis()
        ax.plot(val_loss, label="Validation Loss", **DrawEnum.Style_Plot_Red_H)
        ax.plot(train_loss, label="Train Loss", **DrawEnum.Style_Plot_Black_H)
        ax.set_ylim(-0.05, 1.1)
        ax.set_xlim(-1, self.getInfo("Epoch_Num") + 1)
        plt.xticks(self.getRange(0, self.getInfo("Epoch_Num"), self.getInfo("Epoch_Num") // 10))
        plt.yticks(self.getRange(0, 1, 0.2))
        self.pltShow(title="Loss", footer=None, xLabel='Epoch', yLabel='Loss')


class AnalyzerTrainIter(AnalyzerTrain):

    def __init__(self, res_path):
        super(AnalyzerTrainIter, self).__init__(res_path)

    def getAcc(self, type="train"):
        assert type in ("train", "val"), "type must be train or val"
        if type == "train":
            iter_path = os.path.join(self.path, "train_iter")
        else:
            iter_path = os.path.join(self.path, "val_iter")
        accs = self.readDataFromFile(iter_path, DrawEnum.Iter_IDX_Acc)
        return accs

    def getAvgLoss(self, type="train"):
        assert type in ("train", "val"), "type must be train or val"
        if type == "train":
            iter_path = os.path.join(self.path, "train_iter")
        else:
            iter_path = os.path.join(self.path, "val_iter")
        avg_loss = self.readDataFromFile(iter_path, DrawEnum.Iter_IDX_Loss)
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
        self.pltShow(title="Iteration Loss")

        ax = self.getAxis()
        ax.scatter(train_x, train_acc, s=10, color="black", marker=".", label="Train Accurate", )
        ax.scatter(val_x, val_acc, s=10, color="red", marker=".", label="Val Accurate", )
        ax.set_ylim(-0.05, 1.1)
        # self.pltShow(title="Iteration Accurate", save=f"figs/iter_acc_{self.path.split('/')[-1]}.png")
        self.pltShow(title="Iteration Accurate",)
