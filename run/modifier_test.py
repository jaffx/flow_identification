import sys

sys.path.append(".")
import xlib.modifier.modepoch as mod

total_epoch = 50
modifier = mod.ModifierEpoch("lr*0.9>20", total_epoch=total_epoch)
# modifier.getModStr('show_json')
lr = 1
for epoch in range(total_epoch):
    lr = modifier.mod_lr(epoch, lr, )

