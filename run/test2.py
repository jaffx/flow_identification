import sys
sys.path.append(".")
from xlib.conf import conf
from xlib.modifier.modepoch import ModifierEpoch
from analysis.analyzer import Analyzer

def main():

    modifier = ModifierEpoch("test")
    modifier.getModStr("show")
    modifier.mod_lr(40, 1)


if __name__ == "__main__":
    main()
