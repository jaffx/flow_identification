from model.net import Res1D
from model.net.n2d import Resmodel
import torch
from thop import profile
import json
result = {}


def getNetParamNum1D(name, net):
    data = torch.randn(1, 1, 224)
    flops, params = profile(net, inputs=(data,))
    result[name] = (f"{flops/1e6:.2f}m", f"{params/1e6:.2f}m")


def getNetParamNum2D(name, net):
    data = torch.randn(1, 1, 224, 224)
    flops, params = profile(net, inputs=(data,))
    result[name] = (f"{flops/1e6:.2f}m", f"{params/1e6:.2f}m")


getNetParamNum1D("1d18", Res1D.resnet1d18(include_top=False))
getNetParamNum1D("1d34", Res1D.resnet1d34(include_top=False))
getNetParamNum1D("1d50", Res1D.resnet1d50(include_top=False))
getNetParamNum1D("1d101", Res1D.resnet1d101(include_top=False))

getNetParamNum2D("18", Resmodel.resnet18(include_top=False))
getNetParamNum2D("34", Resmodel.resnet34(include_top=False))
getNetParamNum2D("50", Resmodel.resnet50(include_top=False))
getNetParamNum2D("101", Resmodel.resnet101(include_top=False))

print(json.dumps(result,indent=4))
