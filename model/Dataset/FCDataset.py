import time
import os
import random
import lib.xyq.printer as printer
import lib.xyq.format as formatter
from . import *
import re

DATASET_READ_FINISHED = None


class FCDataset:
    glPattern = re.compile(r"G(?P<gas>[\.\d]+).*L(?P<liquid>[\.\d]+)")
    lgPattern = re.compile(r"L(?P<liquid>[\.\d]+).*G(?P<gas>[\.\d]+)")

    @staticmethod
    def parseGL(fileName: str):
        ret1 = FCDataset.glPattern.match(fileName)
        ret2 = FCDataset.lgPattern.match(fileName)
        ret = ret1 if ret1 else ret2
        if not ret:
            return None
        gas = float(ret.group("gas"))
        liquid = float(ret.group("liquid"))
        return gas, liquid

    def __init__(self, path, length, step, name="Nameless"):
        self.datas = []
        self.path = path
        self.classes = []

        self.step = step
        self.length = length
        if path is not None:
            self.loadDataset(path)
        self.name = name if name else path

    def __len__(self):
        return sum([len(d) for d in self.datas])

    def loadDataset(self, dataset_path):
        """
        从dataset_path加载数据集
        :param dataset_path:
        :return:
        """
        dataset_path = os.path.join(os.getcwd(), dataset_path)
        assert os.path.exists(dataset_path), f"The dataset path \"{dataset_path}\" not exist!"
        time0 = time.time()
        self.path = dataset_path
        class_paths = os.listdir(dataset_path)
        self.classes = class_paths
        total_files = self.getTotalFile()
        suc_count = 0
        fail_count = 0
        # 数据集格式为B格式，即 数据集-train/val-classname-samples
        for cls in class_paths:
            if cls.startswith("."):
                continue
            # 数据集的类名应该是整数类型
            cls_path = os.path.join(dataset_path, cls)
            files = os.listdir(cls_path)
            for file in files:
                if file.startswith("."):
                    continue
                file_path = os.path.join(cls_path, file)
                gas, liquid = self.parseGL(file)
                with open("gs.txt", "a") as fp:
                    fp.write(f"{file}, {gas}, {liquid}\n")
                fdata = flowData.readSimpleData(data_path=file_path, label=(gas, liquid))
                if fdata is None:
                    fail_count += 1
                    printer.xprint_red(f"dataset Error, path {file_path}")
                    continue
                self.datas.append(fdata)
                print(f"\rLoad dataset: 【{suc_count}/{total_files}】| Loading->{file}", end='')
                suc_count += 1
        if fail_count:
            printer.xprint_red(f"\r {fail_count}/{total_files} files load Failed! Please check it!")
            exit(1)
        else:
            printer.xprint_green(
                f"\r【{total_files} Finished】｜Load {dataset_path} running time: {int((time.time() - time0) * 1000)}ms")
        random.shuffle(self.datas)

    """
    @brief 设置Dataset内容
    """

    def setDataset(self, fDataList, path, classes="Unknown"):
        self.datas = fDataList
        self.path = path
        self.classes = classes

    def Init(self):
        for d in self.datas:
            d.Init()

    def isReadable(self):
        return len([i for i in range(len(self.datas)) if self.datas[i].isReadableForLength(self.length)]) > 1

    def getDataProcessRate(self):
        read = sum([d.r_ptr for d in self.datas])
        return read / len(self)

    def getData(self, batch_size):
        readables = [i for i in range(len(self.datas)) if self.datas[i].isReadableForLength(self.length)]
        if readables:
            datas, labels, paths = [], [], []
            for i in range(min(len(readables), batch_size)):
                idx = random.choice(readables)
                readables.remove(idx)
                data, label, path = self.datas[idx].getSample(length=self.length, step=self.step)

                datas.append([data])
                labels.append(label)
                paths.append(path)

            return datas, labels, paths
        else:
            self.Init()
            return DATASET_READ_FINISHED

    def getDPRate(self):
        """
        获取数据处理率(DPRate)
        """
        return sum([d.r_ptr for d in self.datas]) / len(self)

    def getDatasetInfoDict(self):
        dataset_info = {}
        dataset_info["Dataset_Path"] = self.path
        dataset_info["Class_Num"] = len(self.classes)
        dataset_info["Step"] = self.step
        dataset_info["Sample_Length"] = self.length
        return dataset_info

    def getDatasetInfo(self, show=True) -> list:
        """
        获得当前数据集的文本格式信息
        :return: str组成的list
        """
        infos = []
        infos.append(f"Dataset Path:\t{self.path}")
        infos.append(f"Class num:\t{len(self.classes) if self.classes else 'unset'}")
        infos.append(f"Step:{self.step} \t Sample_length {self.length}")
        if show:
            for info in infos:
                printer.xprint(info)
        return infos

    def getTotalFile(self):
        dataset_path = self.path
        total_files = 0
        class_paths = os.listdir(dataset_path)
        for cls in class_paths:
            if cls.startswith("."):
                continue
            cls_path = os.path.join(dataset_path, cls)
            if not os.path.isdir(cls_path):
                continue
            total_files += len(os.listdir(cls_path))
        return total_files
