from tools.run import train
from model.Res1D import resnet1d34

setting = train.train_setting("settings/train/mac.yaml")
print(setting)
model = resnet1d34(4)
train.train(setting, model)
