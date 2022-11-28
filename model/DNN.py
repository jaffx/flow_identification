import torch.nn as nn
import torch

setting1 = [
    (1024, 0.2),
    (512, 0.3),
    (256, 0.4),
    (128, 0.5),
    (4, 1.0),
]
setting2 = [
    (2048, 0.2),
    (2048, 0.2),
    (1024, 0.2),
    (1024, 0.2),
    (4, 1.0),
]


class DNN(nn.Module):
    def __init__(self, setting, input_nums):
        super(DNN, self).__init__()
        nets = []
        for i in range(len(setting)-1):
            in_nums, dr = setting[i]
            out_nums = setting[i + 1][0]
            nets.append(nn.Linear(in_features=in_nums, out_features=out_nums))
            nets.append(nn.Dropout(dr))
        self.fl = nn.Linear(input_nums, setting[0][0])
        self.nets = nn.Sequential(*nets)

    def forward(self, x):
        x = self.fl(x)
        return self.nets(x)


