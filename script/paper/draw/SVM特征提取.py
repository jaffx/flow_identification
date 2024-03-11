import cv2
import numpy as np


def compute_gradient(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=5)
    return gx, gy


def compute_histogram(gradient, bin_size):
    histogram = np.zeros((9, bin_size))
    for i in range(9):
        for j in range(bin_size):
            histogram[i, j] = np.sum(gradient[i * bin_size:(i + 1) * bin_size, j * bin_size:(j + 1) * bin_size])
    return histogram


def compute_hog(img, bin_size):
    height, width = img.shape[:2]
    gx, gy = compute_gradient(img)
    gradient = np.hstack((gx, gy))
    histogram = compute_histogram(gradient, bin_size)
    return histogram


# 读取图像
img = cv2.imread("/Users/lyn/codes/python/Flow_Identification/Flow_Identification/data/image/lena.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 计算HOG特征
hog = compute_hog(img, 8)
# cv2.imshow("hog", hog)
print(hog)
cv2.imwrite("/Users/lyn/codes/python/Flow_Identification/Flow_Identification/data/image/lena_hog.png",hog)
