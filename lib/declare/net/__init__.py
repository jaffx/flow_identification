from . import function, resnet
from model.net import Res1D

function.addNet("ResNet1D-34", resnet.ResNet1D18, "ClassNum=7的ResNet1D-34模型")
