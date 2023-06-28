import time

import numpy as np

from xlib.Dataset.Dataset import flowDataset, DATASET_READ_FINISHED
from xlib.transforms.BaseTrans import transform_base
from xlib.xyq import x_printer as printer, x_time as xtime


class flowDataLoader():
    def __init__(self, dataset: flowDataset, batch_size: int, transform: transform_base, showInfo: bool = False):

        #######数据分析############
        self.dprate = 0
        self.last_read_time = 0
        self.batch_count = 0
        self.sample_count = 0

        self.dataset = dataset
        self.batch_size = batch_size
        self.transform = transform
        self.showInfo = showInfo

    def getData(self):
        rets = self.dataset.getData(self.batch_size)
        if rets == DATASET_READ_FINISHED:
            if self.showInfo:
                printer.xprint_green(
                    f"\r\033[32mFINISH! Dataset name:【{self.dataset.name}】\tDPR:{int(self.dprate * 100)}%\tBatch_count:{self.batch_count}\tSample_count:{self.sample_count}\033[0m",
                    end='\r')
            return DATASET_READ_FINISHED
        else:
            inter_time = time.time() - self.last_read_time if self.last_read_time != 0 else 0
            self.batch_count += 1
            self.sample_count += len(rets[0])
            self.dprate = self.dataset.getDPRate()
            self.last_read_time = time.time()
            estimate_run_time = inter_time * ((self.batch_count / self.dprate) - self.batch_count)
            if self.showInfo:
                printer.xprint(
                    f"\rDataset name:【{self.dataset.name}】\tDPR:{int(self.dprate * 100)}%\tBatch_count:{self.batch_count}\t"
                    f"Sample_count:{self.sample_count}\tRemaining time:{xtime.secsToStr(int(estimate_run_time))}",
                    end='')
            datas, labels, paths = rets
            datas = np.array(datas)
            return self.transform(datas), labels, paths

    def getReadable(self):
        return self.dataset.getReadable()

    def Init(self):
        self.dprate = 0
        self.batch_count = 0
        self.sample_count = 0
        self.last_read_time = 0
        self.dataset.Init()
