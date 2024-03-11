import numpy as np
import matplotlib.pyplot as plt


# 定义sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 定义tanh函数
def tanh(x):
    return np.tanh(x)


# 定义ReLU函数
def relu(x):
    return np.maximum(0, x)


# 生成x轴的值
x = np.linspace(-2, 2, 100)

# 计算每个函数的值
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)
y_relu = relu(x)

# 绘制sigmoid函数图像
plt.figure(figsize=(6, 4))
plt.plot(x, y_sigmoid, label='sigmoid')
plt.plot(x, y_tanh, label='tanh')
plt.plot(x, y_relu, label='ReLU')
plt.legend()
plt.show()
