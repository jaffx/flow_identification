import time
import os
import copy
import random
import tools.xyq.x_printer as printer
import tools.xyq.x_formatter as formatter


class flowData:
    """
    保存一条流型数据
    """

    def __init__(self, data, label, file_path):
        self.r_ptr = 0
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


def readWMSFile(data_path, cls_name) -> flowData | None:
    """
    从文件中加载WMS数据，适用于.epst文件
    @param dataset_path: 数据集地址
    @param cls_name: 类名
    @param filename: 文件名
    @return
    """
    file_path = data_path
    try:

        with open(file_path) as fp:
            content = fp.readlines()
            content = content[2:]
            data = []
            for line in content:
                line = line.strip()
                items = line.split(" ")
                data.append(float(items[-1]))
        return flowData(data, cls_name, file_path)
    except Exception as e:
        printer.xprint_red(f"{file_path} 加载错误，原因 {e}")
        return None


def readSimpleDataset(data_path, cls_name) -> flowData | None:
    file_path = data_path
    try:

        with open(file_path) as fp:
            content = fp.readlines()
            content = content[2:]
            data = []
            for line in content:
                data.append(float(line))
        return flowData(data, cls_name, file_path)
    except Exception as e:
        printer.xprint_red(f"{file_path} 加载错误，原因 {e}")
        return None


DATASET_READ_FINISHED = None


class flowDataset:

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

    def getDatasetInfoDict(self):
        dataset_info = {}
        dataset_info["Dataset_Path"] = self.path
        dataset_info["Class_Num"] = len(self.classes)
        dataset_info["Step"] = self.step
        dataset_info["Sample_Length"] = self.length
        cls_info = {}
        ndata_totals = 0
        nfile_totals = 0
        for fd in self.datas:
            cls_info.setdefault(fd.label, {"Data_Amount": 0, "File_Amount": 0})
            cls_info[fd.label]["Data_Amount"] += len(fd)
            cls_info[fd.label]["File_Amount"] += 1
            ndata_totals += len(fd)
            nfile_totals += 1
        cls_info["Total"] = {"Data_Amount": ndata_totals, "File_Amount": nfile_totals}
        for cls in cls_info:
            cls_info[cls]["Data_Amount"] = formatter.xNumFormat(cls_info[cls]["Data_Amount"], keep_float=1)
            cls_info[cls]["File_Amount"] = cls_info[cls]["File_Amount"]
        dataset_info["Class_Info"] = cls_info
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
        cls_ndata = {}
        cls_nfile = {}
        ndata_totals = 0
        nfile_totals = 0
        for fd in self.datas:
            cls_ndata.setdefault(fd.label, 0)
            cls_nfile.setdefault(fd.label, 0)
            cls_ndata[fd.label] += len(fd)
            cls_nfile[fd.label] += 1
            ndata_totals += len(fd)
            nfile_totals += 1
        infos.append("--------*Dataset Infos*---------")
        infos.append(f"{'label':<8}\t|\t{'Amount':<8}\t|\t{'File':<8}")
        for cls in cls_ndata:
            infos.append(
                f"{cls if cls else 'Unknown':<8}\t|\t {str(int(cls_ndata[cls] / 1000)) + 'k':<8}\t|\t{cls_nfile[cls]:<8}")
        infos.append(f"total\t|\t{int(ndata_totals / 1000)}k\t|\t{nfile_totals}")
        infos.append("*-------*-----------*---------*")
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
                fdata = readSimpleDataset(dataset_path, cls)
                if not fdata:
                    fail_count += 1
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

    def getReadable(self):
        return len([i for i in range(len(self.datas)) if self.datas[i].isReadableForLength(self.length)]) > 0

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
        '''
        获取数据处理率(DPRate)
        :return:
        '''
        return sum([d.r_ptr for d in self.datas]) / len(self)
