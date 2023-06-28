import sys

sys.path.append(".")
from xlib.conf import conf
from xlib.modifier.modepoch import ModifierEpoch
from analysis.analyzer import Analyzer

def main():
    path = "/ex_result/bk/20230702.123500_ResNet1d"
    print(Analyzer(path).checkResult())

if __name__ == "__main__":
    main()
