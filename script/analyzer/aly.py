import sys

sys.path.append('.')
from model.analyzer.AnalyzerTrain import AnalyzerTrainIter, AnalyzerTrainEpoch


# dir_path = "/Users/lyn/codes/python/Flow_Identification/Flow_Identification/ex_result/train"
# for path in os.listdir(dir_path):
#     path = os.path.join(dir_path,path)
#     AnalyzerTrainIter(path).do_aly()

def doaly(path):
    AnalyzerTrainEpoch(path).do_aly()
    AnalyzerTrainIter(path).do_aly()


paths = [
    # "/Users/lyn/codes/python/Flow_Identification/Flow_Identification/ex_result/train/20230704.160806_ResNet1d",
    "/Users/lyn/codes/python/Flow_Identification/Flow_Identification/ex_result/train/20230705.045144_ResNet1d",
    # "/Users/lyn/codes/python/Flow_Identification/Flow_Identification/ex_result/train/20230705.052456_ResNet1d"
    # "/Users/lyn/codes/python/Flow_Identification/Flow_Identification/ex_result/train/20231121.050411_MHNet",
    # "/Users/lyn/codes/python/Flow_Identification/Flow_Identification/ex_result/train/20231120.134503_MHNet",
]

for path in paths:
    doaly(path)
