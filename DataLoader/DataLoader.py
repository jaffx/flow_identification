import copy
import time

from DataLoader.Dataset import flowDataset, DATASET_READ_FINISHED
import threading
from xyq import x_time as xtime


class flowDataLoader():
    def __init__(self, dataset: flowDataset, batch_size: int, transform, showInfo=False):

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
            return DATASET_READ_FINISHED
        else:
            inter_time = time.time() - self.last_read_time if self.last_read_time != 0 else 0
            self.batch_count += 1
            self.sample_count += len(rets[0])
            self.dprate = self.dataset.getDPRate()
            self.last_read_time = time.time()
            estimate_run_time = inter_time * ((self.batch_count / self.dprate) - self.batch_count)
            if self.showInfo:
                print(
                    f"\rDataset name:【{self.dataset.name}】\tDPR:{int(self.dprate * 100)}%\tBatch_count:{self.batch_count}\t"
                    f"Sample_count:{self.sample_count}\tRemaining time:{xtime.secsToStr(int(estimate_run_time))}",
                    end='')
            datas, labels, paths = rets
            return self.transform(datas), labels, paths

    def getReadable(self):
        return self.dataset.getReadable()
