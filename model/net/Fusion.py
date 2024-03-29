from . import *
from . import Module


class FusionBlock(torch.nn.Module):
    """
    特征融合模块，将两个特征向量进行融合
    """

    def __init__(self, in_length1=512, in_length2=512, out_length=512):
        """
        :param in_length1: 特征向量1的长度
        :param in_length2: 特征相连2的长度
        :param out_length: 输出特征向量的长度
        """
        self.toVector = Module.ToVector()
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

    def forward(self, feature1, feature2):
        # 特征1 resize
        feature1 = self.toVector(feature1)
        feature1 = self.resizer1(feature1)
        feature1 = self.bn1(feature1)
        # 特征2 resize
        feature2 = self.toVector(feature2)
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
    特征融合模块，将两个特征向量进行融合，加入残差结构
    """

    def __init__(self, in_length1=512, in_length2=512, out_length=512):
        """
        :param in_length1: 特征向量1的长度
        :param in_length2: 特征相连2的长度
        :param out_length: 输出特征向量的长度
        """

        self.in_length1 = in_length1
        self.in_length2 = in_length2
        self.out_length = out_length
        super(ResFusionBlock, self).__init__()
        self.toVector = Module.ToVector()
        self.resizer1 = torch.nn.Linear(in_length1, out_length)
        self.resizer2 = torch.nn.Linear(in_length2, out_length)
        self.bn1 = torch.nn.BatchNorm1d(out_length)
        self.bn2 = torch.nn.BatchNorm1d(out_length)
        self.fusion = torch.nn.Linear(out_length * 2, out_length)
        self.softMax = torch.nn.Softmax(dim=1)
        self.bn3 = torch.nn.BatchNorm1d(out_length)
        self.w1, self.w2 = torch.tensor(1), torch.tensor(1)

    def forward(self, feature1, feature2):
        # 特征1 resize
        feature1 = self.toVector(feature1)
        feature1 = feature1.view(-1, self.in_length1)
        feature1 = self.resizer1(feature1)
        feature1 = self.bn1(feature1)
        # 特征2 resize
        feature2 = self.toVector(feature2)
        feature2 = feature2.view(-1, self.in_length2)
        feature2 = self.resizer2(feature2)
        feature2 = self.bn2(feature2)
        # 特征拼接
        feature = torch.cat((feature1, feature2), dim=1)
        # 特征融合
        feature = self.fusion(feature) + self.w1 * feature1 + self.w2 * feature2
        feature = self.bn3(feature)
        return feature


class ConvFusion(torch.nn.Module):
    """
    通过卷积操作将两个数据进行融合
    """
    def __init__(self, in_channels1=512, in_channels2=512, out_length=512):
        super().__init__()
        self.in_length1 = in_channels1
        self.in_length2 = in_channels2
        self.out_length = out_length
        self.conv = torch.nn.Conv1d(in_channels=in_channels1 + in_channels2, out_channels=out_length, kernel_size=3)
        self.bn = torch.nn.BatchNorm1d(out_length)
        self.toVector = Module.ToVector()

    def forward(self, feature1, feature2):
        feature = torch.cat((feature1, feature2), dim=1)
        feature = self.conv(feature)
        feature = self.toVector(feature)
        feature = self.bn(feature)
        return feature
