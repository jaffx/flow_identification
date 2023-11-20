import sys

sys.path.append('../../run')
import model.net.MHNet as MHNet
import torch

net = MHNet.MHNet()

input1 = torch.randn((1, 1, 4096))
input2 = torch.randn((1, 1, 4096))

out = net((input1, input2))
print(out)
