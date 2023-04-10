from tools.utils.dataset import *
from tools.utils import setting


def main():
    my_setting = setting.train_setting("../settings/train/mac.yaml")
    print(my_setting)


if __name__ == "__main__":
    main()
