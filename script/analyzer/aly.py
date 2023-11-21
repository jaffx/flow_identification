import sys

sys.path.append('.')
from model.analyzer.epoch import Analyzer_Epoch_Train
from model.analyzer.iter import Analyzer_Iter_Train


# dir_path = "/Users/lyn/codes/python/Flow_Identification/Flow_Identification/ex_result/train"
# for path in os.listdir(dir_path):
#     path = os.path.join(dir_path,path)
#     Analyzer_Iter_Train(path).do_aly()

def doaly(path):
    Analyzer_Epoch_Train(path).do_aly()
    Analyzer_Iter_Train(path).do_aly()


paths = [
    "/Users/lyn/codes/python/Flow_Identification/Flow_Identification/ex_result/train/20230704.160806_ResNet1d",
    # "/Users/lyn/codes/python/Flow_Identification/Flow_Identification/ex_result/train/20230705.045144_ResNet1d",
    # "/Users/lyn/codes/python/Flow_Identification/Flow_Identification/ex_result/train/20230705.052456_ResNet1d"
]

for path in paths:
    doaly(path)
