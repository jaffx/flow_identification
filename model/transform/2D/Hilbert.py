# from pylab import *
import numpy as np


class point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def rotate90(self, x0, y0):
        x2 = y0 + x0 - self.y
        y2 = self.x - x0 + y0
        self.x = x2
        self.y = y2

    def set(self, x, y):
        self.x, self.y = x, y

    def rotate270(self, x0, y0):
        x2 = x0 + self.y - y0
        y2 = x0 - self.x + y0
        self.x = x2
        self.y = y2


def getHilbert(n):
    l = 2 ** n
    points = getpoints(2 ** n, 2 ** n)
    hil = getpoints(1, l * l)
    hil = Hilbert(points, hil, n, 0, 0)
    return hil


def printpoints(points):
    h, w = points.shape
    for i in range(h):
        for j in range(w):
            print("(%s, %s)" % (points[i, j].x, points[i, j].y), end="")
        print("\n", end="")


def Hilbert(points, hil, n, sx, sy):
    l = 2 ** n
    div1, div2, div3 = int(0.25 * l * l), int(l * l / 2), int(l * l * 0.75)
    hl = int(l / 2)
    if n > 1:
        hil[0, 0:div1] = Hilbert(points[0:hl, 0:hl], hil[:, 0:div1], n - 1, sx, sy)[0, ::-1]
        hil[0, div1:div2] = Hilbert(points[0:hl, hl:l], hil[:, div1:div2], n - 1, sx, sy + hl)
        hil[0, div2:div3] = Hilbert(points[hl:l, hl:l], hil[:, div2:div3], n - 1, sx + hl, sy + hl)
        hil[0, div3:l * l] = Hilbert(points[hl:l, 0:hl], hil[:, div3: l * l], n - 1, sx + hl, sy)[0, ::-1]
        for i in range(0, div1):
            hil[0, i].rotate270((sx + sx + hl - 1) / 2, (sy + sy + hl - 1) / 2)
        for i in range(div3, l * l):
            hil[0, i].rotate90((sx + hl + sx + l - 1) / 2, (sy + sy + hl - 1) / 2)
    elif n == 1:
        hil[0] = [points[0, 0], points[0, 1], points[1, 1], points[1, 0]]
    return hil


def getpoints(h, w):
    return np.array([[point(i, j) for j in range(w)] for i in range(h)])


def HilbertFlatten(img, n):
    hil = getHilbert(n)
    result = []
    for p in hil[0]:
        x, y = int(p.x), int(p.y)
        result.append(img[x][y])
    return result


def HilbertBuild(list, n):
    hil = getHilbert(n)
    width = 2 ** n
    result = [[[0, 0, 0] for j in range(width)] for i in range(width)]
    for i in range(len(hil[0])):
        x, y = int(hil[0][i].x), int(hil[0][i].y)
        result[x][y] = list[i]
    return result


def HilbertBuild2(list, hil, n):
    width = 2 ** n
    result = [[[] for j in range(width)] for i in range(width)]
    for i in range(len(hil[0])):
        x, y = int(hil[0][i].x), int(hil[0][i].y)
        result[x][y] = list[i]
    return result


def main():
    result = [[[0, 0, 0] for j in range(256)] for i in range(256)]
    print(len(result), len(result[0]), len(result[0][0]))
