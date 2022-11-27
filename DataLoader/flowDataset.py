import time
import os
import copy
import random
import torch.cuda
import xyq.x_printer as printer


class flowData:
    file_path = ""
    data = []
    r_ptr = 0
    label = None

    def __init__(self, data, label, file_path):
        self.data = data
        self.label = label
        self.file_path = file_path

    def __len__(self):
        return len(self.data)

    def __call__(self, step):
        return self.getSample(step)

    def getSample(self, length, step):
        assert length <= len(self.data), f"The length of sample {length} must less than flow data {len(self.data)}"
        assert step > 0, f"The step {step} is invalid which must greater than 0"
        end = self.r_ptr + length
        if end >= len(self.data):
            r_ptr = 0
        data = copy.deepcopy(self.data[self.r_ptr:end])
        self.r_ptr += step
        return data, self.label, self.file_path

    def Init(self):
        self.r_ptr = 0

    def isReadableForLength(self, length):
        return self.r_ptr + length < len(self.data)


def readWMSFile(dataset_path, cls_name, filename) -> flowData:
    """
    从文件中加载WMS数据，适用于.epst文件
    .epst文件前两为表头和单位，每行前9各字符为时间
    :param dataset_path: 数据集地址
    :param cls_name: 类名
    :param filename: 文件名
    :return:
    """
    file_path = os.path.join(dataset_path, cls_name, filename)
    try:

        with open(file_path) as fp:
            content = fp.readlines()[2:]
            data = [float(line[11:]) for line in content]
        return flowData(data, int(cls_name), file_path)
    except Exception as e:
        printer.xprint_red(f"{file_path} 加载错误，原因 {e}")

        return None


class flowDataset:
    datas = []
    path = ""
    classes = []
    length = 0
    step = 0
    name = ""

    def __init__(self, path, length, step, name=""):
        self.step = step
        self.length = length
        self.loadDataset(path)
        self.name = name if name else path

    def __len__(self):
        return sum([len(d) for d in self.datas])

    def getDatasetInfo(self, show=True) -> list:
        """
        获得当前数据集的文本格式信息
        :return: str组成的list
        """
        infos = []
        infos.append(f"Dataset Path:\t{self.path}")
        infos.append(f"Class num:\t{len(self.classes)}")
        infos.append(f"Step:{self.step} \t Sample_length {self.length}")
        cls_ndata = {}
        totals = 0
        for fd in self.datas:
            cls_ndata.setdefault(fd.label, 0)
            cls_ndata[fd.label] += len(fd)
            totals += len(fd)
        infos.append("------*data_amount*-------")
        infos.append("label\t|\tdata amount")
        for cls in cls_ndata:
            infos.append(f"{cls}\t\t|\t {int(cls_ndata[cls] / 1000)}k")
        infos.append(f"total\t|\t{int(totals / 1000)}k")
        infos.append("------*-----------*-------")
        if show:
            for info in infos:
                printer.xprint(info)
        return infos

    def loadDataset(self, dataset_path):
        '''
        从dataset_path加载数据集
        :param dataset_path:
        :return:
        '''
        dataset_path = os.path.join(os.getcwd(), dataset_path)
        assert os.path.exists(dataset_path), f"The dataset path \"{dataset_path}\" not exist!"
        time0 = time.time()
        self.path = dataset_path
        class_paths = os.listdir(dataset_path)
        self.classes = class_paths
        total_files = sum([len(os.listdir(os.path.join(dataset_path, f))) for f in class_paths])
        suc_count = 0
        fail_count = 0
        for cls in class_paths:
            files = os.listdir(os.path.join(dataset_path, cls))
            for file in files:
                fdata = readWMSFile(dataset_path, cls, file)
                if not fdata:
                    fail_count += 1
                    continue
                self.datas.append(fdata)
                print(f"\rLoad dataset: 【{suc_count}/{total_files}】| Loading->{file}", end='')
                suc_count += 1
        if fail_count:
            printer.xprint_red(f"\r {fail_count}/{total_files} files load Failed! Please check it!")
        else:
            printer.xprint_green(
                f"\r【{total_files} Finished】｜Load {dataset_path} timeout: {int((time.time() - time0) * 1000)}ms")
        random.shuffle(self.datas)

    def Init(self):
        for d in self.datas:
            d.Init()

    def getReadable(self):
        return len([i for i in range(len(self.datas)) if self.datas[i].isReadableForLength(self.length)]) > 0

    def getDataProcessRate(self):
        read = sum([d.r_ptr for d in self.datas])
        return read / len(self)

    def getData(self, batch_size):
        readables = [i for i in range(len(self.datas)) if self.datas[i].isReadableForLength(self.length)]
        if readables:
            ret = []
            for i in range(len(readables)):
                idx = random.choice(readables)
                readables.remove(idx)
                idx_data = self.datas[idx].getSample(length=self.length, step=self.step)
                ret.append(idx_data)
            return ret
        else:
            self.Init()
            return None


train_set = flowDataset(path="../../FlowDataset/Datas2/val", length=128 * 128, step=128 * 64, name="Val Set")
train_set.getDatasetInfo()
dprate = 0
while train_set.getReadable():
    data = train_set.getData(8)
    dprate = train_set.getDataProcessRate()
    print(f"\rDataset Read Process: {int(dprate * 100)}%", end='')
printer.xprint_green(f"\r{train_set.name} process finished! Data processed rate {int(dprate * 100)}%!")
