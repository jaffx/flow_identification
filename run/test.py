import os
import sys

sys.path.append('.')
from analysis.epoch import Analyzer_Epoch_Train
from analysis.iter import Analyzer_Iter_Train

# dir_path = "/Users/lyn/codes/python/Flow_Identification/Flow_Identification/ex_result/train"
# for path in os.listdir(dir_path):
#     path = os.path.join(dir_path,path)
#     Analyzer_Iter_Train(path).do_aly()

def doaly(path):
    # Analyzer_Epoch_Train(path).do_aly()
    Analyzer_Iter_Train(path).do_aly()


paths = [
    "/Users/lyn/codes/python/Flow_Identification/Flow_Identification/ex_result/train/20230626.043853_ResNet1d",
    "/Users/lyn/codes/python/Flow_Identification/Flow_Identification/ex_result/train/20230626.050944_ResNet1d",
    "/Users/lyn/codes/python/Flow_Identification/Flow_Identification/ex_result/train/20230626.053853_ResNet1d",
    "/Users/lyn/codes/python/Flow_Identification/Flow_Identification/ex_result/train/20230626.042137_ResNet1d"
]

for path in paths:
    doaly(path)