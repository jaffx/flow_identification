import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# 创建数据点
np.random.seed(0)
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
Y = [0] * 20 + [1] * 20

# 使用SVM训练模型
clf = svm.SVC(kernel='linear')
clf.fit(X, Y)

# 绘制数据点和分隔超平面
plt.scatter(X[:20, 0], X[:20, 1], color='darkorange', label='A')
plt.scatter(X[20:, 0], X[20:, 1], color='green', label='B')
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# 将超平面绘制在图中
print( clf.coef_)
a = clf.coef_[0] / (clf.coef_[0][0] ** 2 + clf.coef_[0][1] ** 2)
print(a)
xx = np.linspace(xlim[0], xlim[1], 30)
yy = a[0] * xx + a[1]
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM Classification')
plt.legend()
plt.show()
