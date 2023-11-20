import torch.nn as nn


def r_conv(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 1), stride=stride,
                  padding=(1, 0)),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )


def c_conv(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 3), stride=stride,
                  padding=(0, 1)),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )


class RCNet(nn.Module):
    def __init__(self, module_list):
        super(RCNet, self).__init__()
        self.s1 = self.make_layer(module_list[0])
        self.s2 = self.make_layer(module_list[1])
        self.s3 = self.make_layer(module_list[2])
        self.s4 = self.make_layer(module_list[3])
        self.s5 = self.make_layer(module_list[4])
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(module_list[5][0], module_list[5][1])

    def make_layer(self, layer_list):
        model = nn.Sequential()
        for index, c in enumerate(layer_list):
            if c[0] == 'r':
                model.add_module("r_conv{}".format(index), r_conv(c[1], c[2], c[3]))
            elif c[0] == 'c':
                model.add_module("c_conv{}".format(index), c_conv(c[1], c[2], c[3]))
        return model

    def forward(self, x):
        out = self.s1(x)
        out = self.s2(out)
        out = self.s3(out)
        out = self.s4(out)
        out = self.s5(out)
        out = self.pooling(out)
        out = self.fc(out)
        return out
default_module_list = [
    [['r', 3,]]
]
def RCNet_Default():
    return RCNet()