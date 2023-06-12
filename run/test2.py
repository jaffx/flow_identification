import time

from lib.transforms.DataAugmentation import *
from lib.transforms.BaseTrans import *

import numpy as np


def main():
    transform = transfrom_set([
        random_trigger(prob=0.5, transform=transform_selector(transforms = [
            random_noise(),
            normalized_random_noise(),

        ])),
        toTensor()
    ])

    x = np.random.random((32, 1, 16384))
    t0 = time.time()
    x = transform(x)
    t1 = time.time()
    print(f"{int((t1 - t0) * 1000)}ms")
    print(str(transform))


if __name__ == "__main__":
    main()
