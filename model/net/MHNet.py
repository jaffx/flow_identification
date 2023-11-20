import model.net.Res1D as Res1D
import torch


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
        self.avgPool = torch.nn.AdaptiveAvgPool1d(1)
        self.resizer1 = torch.nn.Linear(in_length1, out_length)
        self.resizer2 = torch.nn.Linear(in_length2, out_length)
        self.fusion = torch.nn.Linear(out_length * 2, out_length)
        self.softMax = torch.nn.Softmax(dim=1)

    def __call__(self, feature1, feature2):
        # 特征1 resize
        feature1 = self.avgPool(feature1)
        feature1 = feature1.view(-1, self.in_length1)
        feature1 = self.resizer1(feature1)
        feature1 = self.softMax(feature1)
        # 特征2 resize
        feature2 = self.avgPool(feature2)
        feature2 = feature2.view(-1, self.in_length2)
        feature2 = self.resizer2(feature2)
        feature2 = self.softMax(feature2)
        # 特征拼接
        feature = torch.cat((feature1, feature2), dim=1)
        # 特征融合
        feature = self.fusion(feature)
        feature = self.softMax(feature)
        return feature


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


# 多头分类器
class MHNet(torch.nn.Module):
    def __init__(self, num_classes=7):
        super(MHNet, self).__init__()
        self.head1 = Res1D.resnet1d34(include_top=False)
        self.head2 = Res1D.resnet1d34(include_top=False)
        self.fusion = FusionBlock(512, 512, 256)
        self.classifier = Classifier(256, num_classes)

    def __call__(self, input: tuple):
        feature1 = self.head1(input[0])
        feature2 = self.head1(input[1])
        feature = self.fusion(feature1, feature2)
        feature = self.classifier(feature)
        return feature
