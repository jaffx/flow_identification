import math
from matplotlib import pyplot as plt
import numpy as np
from DataLoader import Hilbert as hilb


def showHilBert(hil, resolution):
    _, l = hil.shape
    k = int(resolution / math.sqrt(l))
    X = [hil[0, i].x * k + k // 2 for i in range(l)]
    Y = [hil[0, i].y * k + k // 2 for i in range(l)]
    im = (np.zeros((resolution, resolution, 3))+0)/255

    fig = plt.figure(figsize=(8, 8))
    # fig.patch.set_facecolor('yellow')
    ax = fig.add_subplot(111)
    fig.patch.set_alpha(0.5)
    ax.set_xticks(np.arange(0, resolution, k))
    ax.set_yticks(np.arange(0, resolution, k))
    ax.imshow(im)
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    plt.plot(X, Y, 'red')
    plt.grid()
    plt.savefig(f"hils/hilbert_{int(math.log2(math.sqrt(l)))}.png")
    plt.show()


def test():
    fig = plt.figure(figsize=(4, 4))
    # 画布边缘设置颜色
    fig.patch.set_facecolor('yellow')
    # 设置透明度
    fig.patch.set_alpha(0.5)

    # num must be 1 <= num <= 1
    ax = fig.add_subplot(111)
    # 设置背景颜色
    ax.patch.set_facecolor('red')
    # 设置透明度
    ax.patch.set_alpha(0.5)

    x = [1, 2, 3]
    y = [2, 4, 6]
    plt.plot(x, y)
    plt.show()


for i in range(1, 7):
    h = hilb.getHilbert(i)
    showHilBert(h, 512)



