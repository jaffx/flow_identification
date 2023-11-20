import os
import random

import lib.xyq.printer as printer
import lib.xyq.format as formatter
from lib import xyq


class MSFlowData:
    """
    为多数据源设计的流型数据存储结构，用来获取多数据源的数据样本
    通过files传入指定的数据文件，files的长度必须大于等于2
    files[0]作为基准数据（基准文件）
    """

    def __init__(self, files: list, label):
        """
        :param files: list 传入一个文件数组
        :param label: 该样本的标签
        """
        self.files = files
        self.label: int = int(label)
        self.datas = []
        self.r_ptrs = []

        # 保存所有数据的最小数据长度，读取样本长度不能超过该值
        self._minLength = None
        self.loadData()

    def __len__(self):
        return len(self.datas[0])

    def __call__(self):
        return self.getSample

    def getSample(self, length, step):
        assert length <= len(
            self.datas[0]), f"The length of sample {length} must less than flow data {len(self.datas[0])}"
        assert step > 0 and length > 0, f"step and length must greater than 0"
        data = []
        for i in range(len(self.datas)):
            if self.r_ptrs[i] + length >= len(self.datas[i]):
                self.r_ptrs[i] = random.randint(0, length)
            d = self.datas[i][self.r_ptrs[i]:self.r_ptrs[i] + length]
            data.append([d])
            self.r_ptrs[i] += step
        return data, self.label, self.files[0]

    def Init(self):
        self.r_ptrs = [0 for i in range(len(self.datas))]

        self._minLength = min([len(self.datas[i]) for i in range(len(self.datas))])

    def isReadableForLength(self, length):
        """
        判断该数据是否还能读取长度为length的数据，主要是以基准数据为判断依据
        :param length:
        :return:
        """
        return self.r_ptrs[0] + length < len(self.datas[0])

    def getSubPath(self, index=0):
        """
        获取数据文件在数据集文件夹下的相对路径
        :param index:
        :return:
        """
        return f"/{self.label}/{self.files[index]}"

    def loadData(self):
        for file in self.files:
            assert os.path.isfile(file), f"数据文件{file}不存在"
            with open(file) as fp:
                content = fp.readlines()
                data = []
                for d in content:
                    try:
                        data.append(float(d))
                    except Exception as e:
                        print(e)
                        continue
                self.datas.append(data)
        self.Init()


class MSDataset:
    @staticmethod
    def getMappingFile(mappingFile: str) -> list:
        """
        读取数据集映射文件，将内容组织成list返回
            映射文件格式：类别 数据文件1 数据文件2 ...
            文件1是主文件，其他文件为次要文件
        :param mappingFile 映射文件路径
        """
        assert os.path.isfile(mappingFile), f"映射文件路径{mappingFile}不存在"
        with open(mappingFile) as fp:
            content = fp.readlines()
            mapping = []
            for line in content:
                items = line.strip('\n').split("\t")
                assert len(items) >= 3, f"映射文件列数错误"
                cls = items[0]
                files = items[1:]
                mapping.append((cls, files))
            return mapping

    def __init__(self, length, step, mappingFile: str = xyq.const.XYQ_CONF_PATH_MAPPING_FILE, path: str = "",
                 sources: list = None, name=""):
        """
        :type path: str
        :param mappingFile: str 映射文件路径
        """
        self.path = path
        self.mappingFile = mappingFile
        self.dataset = []
        self.length = length
        self.step = step
        self.sources = sources if sources else ["wms", "pressure"]
        self.name = name
        self.__length__ = None
        self.__load()

    def __len__(self):
        if self.__length__ is None:
            self.__length__ = sum([len(d) for d in self.dataset])
        return self.__length__

    def __load(self):
        mapping = MSDataset.getMappingFile(self.mappingFile)
        for cls, files in mapping:
            assert len(files) == len(self.sources)
            absFiles = [os.path.join(self.path, self.sources[i], cls, files[i]) for i in range(len(files))]
            data = MSFlowData(label=cls, files=absFiles)
            self.dataset.append(data)
        random.shuffle(self.dataset)

    def getDatasetInfoDict(self):
        dataset_info = {}
        dataset_info["Step"] = self.step
        dataset_info["Sample_Length"] = self.length
        dataset_info["Dataset_Path"] = self.path
        cls_info = {}
        ndata_totals = 0
        nfile_totals = 0
        for fd in self.dataset:
            cls_info.setdefault(fd.label, {"Data_Amount": 0, "File_Amount": 0})
            cls_info[fd.label]["Data_Amount"] += len(fd)
            cls_info[fd.label]["File_Amount"] += 1
            ndata_totals += len(fd)
            nfile_totals += 1
        cls_info["Total"] = {"Data_Amount": ndata_totals, "File_Amount": nfile_totals}
        for cls in cls_info:
            cls_info[cls]["Data_Amount"] = formatter.xNumFormat(cls_info[cls]["Data_Amount"], keep_float=1)
            cls_info[cls]["File_Amount"] = cls_info[cls]["File_Amount"]
        return dataset_info

    def getDatasetInfo(self, show=True) -> list:
        """
        获得当前数据集的文本格式信息
        :return: str组成的list
        """
        infos = []
        infos.append(f":\t{self.path}")
        infos.append(f"Step:{self.step} \t Sample_length {self.length}")
        cls_ndata = {}
        cls_nfile = {}
        ndata_totals = 0
        nfile_totals = 0
        for fd in self.dataset:
            cls_ndata.setdefault(fd.label, 0)
            cls_nfile.setdefault(fd.label, 0)
            cls_ndata[fd.label] += len(fd)
            cls_nfile[fd.label] += 1
            ndata_totals += len(fd)
            nfile_totals += 1
        infos.append("--------*Dataset Infos*---------")
        infos.append(f"{'label':<8}|{'Amount':<8}|{'File':<8}")
        for cls in cls_ndata:
            infos.append(
                f"{cls if cls is not None else 'Unknown':<8}|{str(int(cls_ndata[cls] / 1000)) + 'k':<8}|{cls_nfile[cls]:<8}")
        infos.append(f"{'total':<8}|{str(int(ndata_totals / 1000)) + 'k':<8}|{nfile_totals:<8}")
        infos.append("*-------*-----------*---------*")
        if show:
            for info in infos:
                printer.xprint(info)
        return infos

    def Init(self):
        for d in self.dataset:
            d.Init()

    def isReadable(self):
        """
        判断数据集是否读完
        :return:
        """
        return len([i for i in range(len(self.dataset)) if self.dataset[i].isReadableForLength(self.length)]) > 0

    def getData(self, batch_size):
        readables = [i for i in range(len(self.dataset)) if self.dataset[i].isReadableForLength(self.length)]
        if readables:
            datas, labels, paths = [[] for _ in range(len(self.sources))], [], []
            for i in range(min(len(readables), batch_size)):
                idx = random.choice(readables)
                readables.remove(idx)
                data, label, path = self.dataset[idx].getSample(length=self.length, step=self.step)
                for src in range(len(datas)):
                    datas[src].append(data[src])
                labels.append(label)
                paths.append(path)

            return datas, labels, paths
        else:
            self.Init()
            return None

    def getDPRate(self):
        """
        获取数据处理率(DPRate)
        """
        return sum([d.r_ptrs[0] for d in self.dataset]) / len(self)
