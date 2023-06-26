import os
import sys
sys.path.append('.')
from analysis.epoch import Analyzer_Epoch_Train
from analysis.iter import Analyzer_Iter_Train

# dir_path = "/Users/lyn/codes/python/Flow_Identification/Flow_Identification/ex_result/train"
# for path in os.listdir(dir_path):
#     path = os.path.join(dir_path,path)
#     Analyzer_Iter_Train(path).do_aly()

path = "/Users/lyn/codes/python/Flow_Identification/Flow_Identification/ex_result/train/20230615.163450 [ResNet1d]"
Analyzer_Epoch_Train(path).do_aly()
Analyzer_Iter_Train(path).do_aly()
