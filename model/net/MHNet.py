from . import *


# 多头分类器
class MSFINet(torch.nn.Module):
    def __init__(self, num_classes=7, head1=None, head2=None, fusion=None):
        super(MSFINet, self).__init__()
        # 特征提取模块1
        if head1 is None:
            self.head1 = Res1D.resnet1d34(include_top=False)
        else:
            self.head1 = head1
        # 特征提取模块2
        if head2 is None:
            self.head2 = Res1D.resnet1d34(include_top=False)
        else:
            self.head2 = head2
        # 特征融合模块
        if fusion is None:
            self.fusion = Fusion.FusionBlock(512, 512, 512)
        else:
            self.fusion = fusion
        self.fusionClassifier = Module.Classifier(512, num_classes)
        self.classifier1 = Module.Classifier(512, num_classes)
        self.classifier2 = Module.Classifier(512, num_classes)
        self.toVector = Module.ToVector()
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def callNet1(self, x: torch.Tensor) -> torch.Tensor:
        """
        通过数据源1进行预测
        @param x:
        @return:
        """
        feature = self.head1(x)
        feature = self.toVector(feature)
        return self.classifier1(feature)

    def callNet2(self, x: torch.Tensor):
        """
        通过数据源2进行预测
        @param x:
        @return:
        """
        feature = self.head2(x)
        feature = self.toVector(feature)
        return self.classifier2(feature)

    def callFusion(self, x):
        """
        使用双数据源融合进行预测
        @param x:
        @return:
        """
        feature1 = self.head1(x[0])
        feature2 = self.head2(x[1])
        feature = self.fusion(feature1, feature2)
        return self.fusionClassifier(feature)

    def callAll(self, x) -> tuple:
        """
        三种途径综合预测
        @param x:
        @return:
        """
        feature1 = self.head1(x[0])
        feature2 = self.head2(x[1])
        feature = self.fusion(feature1, feature2)
        return self.classifier1(feature1), self.classifier2(feature2), self.fusionClassifier(feature)

    def forward(self, x: tuple):
        if x[0] is not None and x[1] is not None:
            return self.callFusion(x)
        elif x[0] is not None:
            return self.callNet1(x[0])
        else:
            return self.callNet2(x[1])
