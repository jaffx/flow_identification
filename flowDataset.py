import time
import os
import copy

import torch.cuda


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
        ret = copy.deepcopy(self.data[self.r_ptr:end])
        self.r_ptr += step
        return ret

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
            # print(f"成功加载文件【{file_path}】,行数{len(content)}")
            data = [float(line[9:]) for line in content]
        return flowData(data, int(cls_name), file_path)
    except Exception as e:
        print(f"{file_path} 加载错误，原因 {e}")
        return None


class flowDataset:
    datas = []
    path = ""
    classes = []

    def getDatasetInfo(self) -> list:
        """
        获得当前数据集的文本格式信息
        :return: str组成的list
        """
        infos = []
        infos.append(f"Dataset Path:\t{self.path}")
        infos.append(f"Class num:\t{len(self.classes)}")
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
        return infos

    def loadDataset(self, dataset_path):
        assert os.path.exists(
            os.path.join(os.getcwd(), dataset_path)), f"The dataset path \"{dataset_path}\" not exist!"
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
            print(
                f"\r\033[31mLoad dataset: {fail_count}/{total_files} files load Failed! Please check it!\033[0m"
                f"\033[0m")
        else:
            print(
                f"\r\033[32mLoad dataset: 【{total_files} Finished】｜Load {dataset_path} timeout: {int((time.time() - time0) * 1000)}ms"
                f"\033[0m")


print(torch.cuda.is_available())
