from . import *


# 分类器
class Classifier(torch.nn.Module):
    def __init__(self, feature_length, class_num):
        super(Classifier, self).__init__()
        self.cls = torch.nn.Linear(feature_length, class_num)
        self.softmax = torch.nn.Softmax(dim=1)

    def __call__(self, x):
        x = self.cls(x)
        x = self.softmax(x)
        return x


class ToVector(torch.nn.Module):
    def __init__(self):
        super(ToVector, self).__init__()
        self.avgPool = torch.nn.AdaptiveAvgPool1d(1)
        pass

    def __call__(self, x):
        batch, channel, _ = x.shape
        x = self.avgPool(x)
        x = x.view(batch, channel)
        return x
