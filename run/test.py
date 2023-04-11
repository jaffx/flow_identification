from tools.utils.dataset import *
from tools.utils import setting
from tools.transforms.Preprocess import *
from matplotlib import pyplot as plt


def main():
    transform = normalization()
    data = np.random.random((1, 1, 1024)) * 100 + 30
    data = np.sin(np.cos(data))
    data = transform(data)
    plt.plot(data[0][0])
    plt.show()
    print(np.mean(data), np.std(data))


if __name__ == "__main__":
    main()
