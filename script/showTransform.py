import random

from DataLoader import transforms
from matplotlib import pyplot as plt
from DataLoader import Hilbert as hilb
import cv2
import numpy as np

start_color = np.array([0, 0, 0])
end_color = np.array([220, 220, 220])
wh = 32 * 2
wh2 = 64
n = 6
dts = [[], [], [], []]
dts[0] = "/Users/lyn/codes/python/Flow_Identification/Dataset/train/0/L300G50_Y_Sensor_2.epst"
dts[1] = "/Users/lyn/codes/python/Flow_Identification/Dataset/train/1/L200G230_Y_Sensor_2.epst"
dts[2] = "/Users/lyn/codes/python/Flow_Identification/Dataset/train/2/L200G400_Y_Sensor_2.epst"
dts[3] = "/Users/lyn/codes/python/Flow_Identification/Dataset/train/3/L100G500_Y_Sensor_2.epst"
svpth = "figs/"
save_path = "figs/"


def read_from_file(file):
    with open(file) as fp:
        content = fp.readlines()[2:]
        data = []
        for i in range(wh * wh):
            data.append(float(content[i][12:-1]))
        return data


def tdata_color(data):
    dt = np.zeros((1, wh * wh, 3))
    for i in range(wh * wh):
        dt[0][i] = start_color + data[i] / 100 * (end_color - start_color)
    return dt


def show_hilbert(datas):
    hil = hilb.getHilbert(n)
    hil_data = hilb.HilbertBuild2(datas[0], hil, n)
    hil_data = np.array(hil_data)
    showdata(hil_data, hil_data.shape, f'{save_path}/hil.jpg')


def show_spf(datas):
    print(datas.shape)
    sd = np.zeros((wh, wh, 3))
    for i in range(wh):
        sd[i] = datas[0][i * wh:(i + 1) * wh]
    # for i in range(3):
    #     sd = np.log1p(sd)
    showdata(sd, sd.shape, f'{save_path}/SpacoiusFloder.jpg')


def origin_data(length, start_color, end_color):
    datas = np.zeros((1, length, 3))
    for i in range(0, length):
        datas[0][i] = start_color + (end_color - start_color) / (length) * i
    return datas


def show_origin(length):
    plt.figure(figsize=(18, 4), dpi=100)
    data = origin_data(length, start_color, end_color) / 255
    plt.imshow(data)
    ax = plt.gca()
    ax.set_xticks(np.arange(-0.5, length + 0.5, 1))
    ax.set_yticks(np.arange(-0.5, 1 + 0.5, 1))
    plt.xlim(-0.5, length - 0.5)
    plt.ylim(-0.5, 1 - 0.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.tick_params(axis='x', colors='white')
    plt.tick_params(axis='y', colors='white')
    # ax.grid(color='white')
    plt.savefig('figs/origin.jpg')
    plt.show()


def showdata(data, shape, save_name):
    sp = shape
    plt.figure(figsize=(200, 200), dpi=10)
    plt.imshow(data)
    ax = plt.gca()
    ax.set_xticks(np.arange(-0.5, sp[0] + 0.5, 1))
    ax.set_yticks(np.arange(-0.5, sp[1] + 0.5, 1))
    plt.xlim(-0.5, shape[0] - 0.5)
    plt.ylim(-0.5, shape[1] - 0.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.tick_params(axis='x', colors='white')
    plt.tick_params(axis='y', colors='white')
    # ax.grid(color='white', linewidth=30)

    if save_name:
        plt.savefig(save_name)
    plt.show()


def show_fft(datas):
    fd = []
    for i in range(wh):
        dt = datas[i * wh:(i + 1) * wh]
        dt = np.array(dt)
        dt = abs(np.fft.fft(dt))
        # print(dt.shape)
        fd.append(dt)
    fd = np.array(fd)
    maps = np.zeros((wh, wh, 3))
    for i in range(wh):
        for j in range(wh):
            maps[i, j] = start_color + (end_color - start_color) / (fd.max() - fd.min()) * fd[i, j]
    showdata(maps / 255, fd.shape, f"{save_path}/fft-{0}.jpg")
    for ii in range(3):
        fd = np.log1p(fd)
        print(fd.std())
        maps = np.zeros((wh, wh, 3))
        for i in range(wh):
            for j in range(wh):
                maps[i, j] = start_color + (end_color - start_color) / (fd.max() - fd.min()) * fd[i, j]
        showdata(maps / 255, fd.shape, f"{save_path}/fft-{ii + 1}.jpg")


# for i in range(4):
#     d = dts[i]
#     save_path = f"{svpth}/{i}/"
#     conf = read_from_file(d)
#     with open(save_path+'conf.txt', 'w+') as fp:
#         for dd in conf:
#             fp.write(f"{dd}\n")
#     tdata = tdata_color(conf) / 255
#     show_fft(conf)
#     show_hilbert(tdata)
#     show_spf(tdata)
show_fft([i for i in range(wh * wh)])
