from .. import Res1D, Module
import torch


class FCRes1D(torch.nn.Module):
    def __init__(self):
        super(FCRes1D, self).__init__()
        self.head = Res1D.resnet1d34(include_top=False)
        self.avgPool = Module.ToVector()
        self.fc1 = torch.nn.Linear(in_features=512, out_features=256)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(in_features=256, out_features=2)

    def __call__(self, x):
        x = self.head(x)
        x = self.avgPool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
