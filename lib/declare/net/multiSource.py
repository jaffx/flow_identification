from . import *
from model.net import MHNet
from model.net import Module, Fusion


def MSNetWithBasicFusion(num_classes=7):
    return MHNet.MSFINet(num_classes=num_classes)


def MSNetWithResFusion(num_classes=7):
    return MHNet.MSFINet(fusion=Fusion.ResFusionBlock(), num_classes=num_classes)


def MSNetWithConvFusion(num_classes=7):
    return MHNet.MSFINet(fusion=Fusion.ConvFusion(), num_classes=num_classes)


def MSNetWithHalfBranch(num_classes=7):
    return MHNet.MSFINet_HalfBranch(num_classes=num_classes)


def MSNetWithFullTrunk(num_classes=7):
    return MHNet.MSFINet_FullTrunk(num_classes=num_classes)
