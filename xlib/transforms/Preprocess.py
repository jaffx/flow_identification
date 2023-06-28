from xlib.transforms import BaseTrans as BT
import numpy as np


class normalization(BT.transform_base):
    def __init__(self):
        super().__init__()

    def __call__(self, x: np.array):
        out = x.copy()
        for i in range(len(out)):
            batch = out[i]
            mean = np.mean(batch)
            std = np.nanstd(batch)
            std = max(std, 0.001)
            out[i] = (batch - mean) / std
        return out


class divide(BT.transform_base):
    def __init__(self, divisor: float):
        super().__init__()
        self.divisor = divisor

    def __call__(self, x: np.array) -> np.array:
        return x / self.divisor
