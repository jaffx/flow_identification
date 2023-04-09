import numpy as np


class normalized_random_noise:
    def __init__(self, mean, sqr):
        self.mean = mean
        self.sqr = sqr


class random_noise:
    def __init__(self, noise_min, noise_max):
        self.noise_min, self.noise_max = noise_min, noise_max

    def __call__(self, x):
        shape = x.shape
        noise = np.random.random(size=shape) * (self.noise_max - self.noise_min) + self.noise_min
        return x + noise
