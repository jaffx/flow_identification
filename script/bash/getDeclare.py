import sys

sys.path.append(".")
from lib.declare import net, transform


def show(func):
    fence = "#" * 60

    def _show_():
        print(fence)
        func()
        print(fence)

    return _show_


def _print(*args):
    print("\t" + "\t".join(args))


@show
def showNet():
    print("神经网络：")
    nets = net.function.getAllNetInfo()
    for n in nets:
        _print(n["name"], n["desc"])


@show
def showTransform():
    print("transform:")
    trans = transform.function.getAllTransformInfos()
    for t in trans:
        _print(t["name"], t["desc"])


showNet()
showTransform()
