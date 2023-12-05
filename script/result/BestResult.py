import os
import shutil
import sys

sys.path.append(".")
from model import analyzer as alyer

rPath = "./ex_result/train"
results = os.listdir(rPath)
accs = []
for r in results:
    try:
        eAlyer = alyer.epoch.Analyzer_TrainEpoch(path=os.path.join(rPath, r))
        acc = eAlyer.getBestAcc()
        accs.append(acc)
    except Exception as e:
        # print(r,e)
        continue
print(accs)
