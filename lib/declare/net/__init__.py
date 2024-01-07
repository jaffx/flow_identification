from . import function, resnet, multiSource
from model.net import Res1D,Module

function.addNet("ResNet1D-34", resnet.ResNet1D18, "ClassNum=7的ResNet1D-34模型")

function.addNet("MSFINet-ResFusion", multiSource.MSNetWithResFusion, "多源网络，使用残差结构做融合")
function.addNet("MSFINet-BasicFusion", multiSource.MSNetWithResFusion, "多源网络，使用基本特征向量基本融合")
