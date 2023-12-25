import time
import numpy as np
from ..transform.BaseTrans import transformBase
from ..Dataset import MSDataset
from lib import xyq


class flowDataLoader():
    def __init__(self, dataset, batch_size: int, transform: transformBase, showInfo: bool = False):
        ####### 数据分析 #######
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
        if rets is None or len(rets[0]) == 1:
            # 如果一个batch只有一个数据，bn层会出错，这种情况下默认训练结束
            if self.showInfo:
                xyq.printer.xprint_green(
                    f"\r\033[32mFINISH! Dataset name:【{self.dataset.name}】    "
                    f"DPR:{int(self.dprate * 100)}%    "
                    f"Batch_count:{self.batch_count}    "
                    f"Sample_count:{self.sample_count}\033[0m",
                    end='\r')
            return None
        else:
            datas, labels, paths = rets
            inter_time = time.time() - self.last_read_time if self.last_read_time != 0 else 0
            self.batch_count += 1
            self.sample_count += len(datas)
            self.dprate = self.dataset.getDPRate()
            self.last_read_time = time.time()
            estimate_run_time = inter_time * ((self.batch_count / self.dprate) - self.batch_count)
            if self.showInfo:
                xyq.printer.xprint_blue(
                    f"\rDataset name:【{self.dataset.name}】    "
                    f"DPR:{int(self.dprate * 100)}%    "
                    f"Batch_count:{self.batch_count}    "
                    f"Sample_count:{self.sample_count}    "
                    f"Remaining time:{xyq.format.secsToStr(int(estimate_run_time))}",
                    end='\r')
            datas = np.array(datas)
            return self.transform(datas), labels, paths

    def isReadable(self):
        return self.dataset.isReadable()

    def Init(self):
        self.dprate = 0
        self.batch_count = 0
        self.sample_count = 0
        self.last_read_time = 0
        self.dataset.Init()
