import torch
import os

import yaml

import xyq.x_time as xtime
from model.model_v2 import MobileNetV2
from DataLoader.transforms import *
from DataLoader.Dataset import flowDataset
from DataLoader.DataLoader import flowDataLoader

# 设置权重文件地址
weight_path = "logs/train_result_path/2022-12-01 15.59.38 [MobileNetV2]/2022-12-01 15.59.38 [MobileNetV2].pth"

# 设置数据集及相关参数
data_length = 128 * 128
sampling_step = 128 * 64
batch_size = 1
val_set_path = "../Dataset/val"
transform = flowHilbertTransform(7)
val_dataset = flowDataset(path=val_set_path, length=data_length, step=sampling_step, name="Validation Set")
val_loader = flowDataLoader(dataset=val_dataset, batch_size=batch_size, transform=transform, showInfo=True)

# 加载模型
device = torch.device(("cuda:0" if torch.cuda.is_available() else "cpu"))
print("Using device {}".format(device))
net = MobileNetV2(4)
# net.load_state_dict(torch.load(weight_path, map_location=device))
loss_function = torch.nn.CrossEntropyLoss()

# 数据保存相关内容
model_name = net.__class__.__name__
task_name = f"[{model_name}]{xtime.getDateTimeForPath()}"
log_path = os.path.join(os.getcwd(), "logs", "val", task_name)
info_path = os.path.join(log_path, "info.yaml")
iter_path = os.path.join(log_path, "iter")
os.makedirs(log_path)

task_info = {
    "Task_Time": xtime.getDateTime(),
    "Model_Name": model_name,
    "Data_Length": data_length,
    "Sampling_Step": sampling_step,
    "Loss_Function": loss_function.__class__.__name__,
    "Transform": transform.__class__.__name__,
    "Model_Parameters_Amount": sum([p.nelement() for p in net.parameters()]),
    "Batch_Size": batch_size,
    "Val_Set_Info": val_dataset.getDatasetInfoDict(),
}


def main():
    # 准备工作
    print("Begin Validation Task!")
    start_time = time.time()

    with open(info_path, "w") as fp:
        yaml.dump(task_info, fp)

    with torch.no_grad():
        while val_loader.getReadable():
            data, label, path = val_loader.getData()
            # data = data.to(device)
            # label = torch.Tensor(label).to(device)
            predict_y = net(data)


if __name__ == "__main__":
    main()
