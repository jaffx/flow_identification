import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('/Users/lyn/codes/python/Flow_Identification/Flow_Identification/data/image/lena.png')

# 转换为灰度图
input_data = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
input_data = cv2.resize(input_data, (50, 50))

k = 2


# 最大池化操作
def maxPooling(img):
    h, w = img.shape
    r = np.zeros((h // k, w // k))
    for i in range(h // k):
        for j in range(w // k):
            r[i, j] = np.max(img[i * k:i * k + k - 1, j * k:j * k + k - 1])
    return r


def avgPooling(img):
    h, w = img.shape
    r = np.zeros((h // k, w // k))
    for i in range(h // k):
        for j in range(w // k):
            r[i, j] = np.mean(img[i * k:i * k + k - 1, j * k:j * k + k - 1])
    return r


plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus'] = False
mp = maxPooling(input_data)
ap = avgPooling(input_data)
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].imshow(input_data, cmap='gray')
axs[0].set_title('原始数据')
axs[0].axis('off')
axs[1].imshow(mp, cmap='gray')
axs[1].set_title('最大池化输出')
axs[1].axis('off')
axs[2].imshow(ap, cmap='gray')
axs[2].set_title('平均池化输出')
axs[2].axis('off')
plt.show()
