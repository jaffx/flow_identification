from . import *


class FusionBlock(torch.nn.Module):
    """
    特征融合模块，将两个特征向量进行融合
    """

    def __init__(self, in_length1, in_length2, out_length):
        """
        :param in_length1: 特征向量1的长度
        :param in_length2: 特征相连2的长度
        :param out_length: 输出特征向量的长度
        """
        self.in_length1 = in_length1
        self.in_length2 = in_length2
        self.out_length = out_length
        super(FusionBlock, self).__init__()
        self.resizer1 = torch.nn.Linear(in_length1, out_length)
        self.resizer2 = torch.nn.Linear(in_length2, out_length)
        self.bn1 = torch.nn.BatchNorm1d(out_length)
        self.bn2 = torch.nn.BatchNorm1d(out_length)
        self.fusion = torch.nn.Linear(out_length * 2, out_length)
        self.softMax = torch.nn.Softmax(dim=1)
        self.bn3 = torch.nn.BatchNorm1d(out_length)

    def __call__(self, feature1, feature2):
        # 特征1 resize
        feature1 = self.resizer1(feature1)
        feature1 = self.bn1(feature1)
        # 特征2 resize
        feature2 = self.resizer2(feature2)
        feature2 = self.bn2(feature2)
        # 特征拼接
        feature = torch.cat((feature1, feature2), dim=1)
        # 特征融合
        feature = self.fusion(feature)
        feature = self.bn3(feature)
        return feature


class ResFusionBlock(torch.nn.Module):
    """
    特征融合模块，将两个特征向量进行融合
    """

    def __init__(self, in_length1, in_length2, out_length):
        """
        :param in_length1: 特征向量1的长度
        :param in_length2: 特征相连2的长度
        :param out_length: 输出特征向量的长度
        """
        self.in_length1 = in_length1
        self.in_length2 = in_length2
        self.out_length = out_length
        super(ResFusionBlock, self).__init__()
        self.avgPool = torch.nn.AdaptiveAvgPool1d(1)
        self.resizer1 = torch.nn.Linear(in_length1, out_length)
        self.resizer2 = torch.nn.Linear(in_length2, out_length)
        self.bn1 = torch.nn.BatchNorm1d(out_length)
        self.bn2 = torch.nn.BatchNorm1d(out_length)
        self.fusion = torch.nn.Linear(out_length * 2, out_length)
        self.softMax = torch.nn.Softmax(dim=1)
        self.bn3 = torch.nn.BatchNorm1d(out_length)

    def __call__(self, feature1, feature2):
        # 特征1 resize
        feature1 = self.avgPool(feature1)
        feature1 = feature1.view(-1, self.in_length1)
        feature1 = self.resizer1(feature1)
        feature1 = self.bn1(feature1)
        # 特征2 resize
        feature2 = self.avgPool(feature2)
        feature2 = feature2.view(-1, self.in_length2)
        feature2 = self.resizer2(feature2)
        feature2 = self.bn2(feature2)
        # 特征拼接
        feature = torch.cat((feature1, feature2), dim=1)
        # 特征融合
        feature = self.fusion(feature)
        feature = self.bn3(feature)
        return feature
