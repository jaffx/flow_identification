from . import *


# 多头分类器
class MHNet(torch.nn.Module):
    def __init__(self, num_classes=7):
        super(MHNet, self).__init__()
        self.head1 = Res1D.resnet1d34(include_top=False)
        self.head2 = Res1D.resnet1d34(include_top=False)
        self.fusion = Fusion.FusionBlock(512, 512, 512)
        self.fusionClassifier = Module.Classifier(512, num_classes)
        self.classifier1 = Module.Classifier(512, num_classes)
        self.classifier2 = Module.Classifier(512, num_classes)

        self.toVector = Module.ToVector()

        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def __call__(self, x: tuple):
        feature1, feature2, cls1, cls2 = None, None, None, None
        if x[0] is not None:
            feature1 = self.head1(x[0])
            feature1 = self.toVector(feature1)
            cls1 = self.classifier1(feature1)
        if x[1] is not None:
            feature2 = self.head2(x[1])
            feature2 = self.toVector(feature2)
            cls2 = self.classifier2(feature2)
        if x[0] is not None and x[1] is not None:
            # 使用融合特征进行预测
            feature = self.fusion(feature1, feature2)
            cls = self.fusionClassifier(feature)
            return (cls + cls1 + cls2) / 3
        return cls1 if x[0] is not None else cls2
