import torch
import os

import yaml
import sys

sys.path.append(".")
import lib.xyq.printer as printer
import lib.xyq.format as formatter
from model.net.Res1D import resnet1d34
from lib.declare import transform
from model.Dataset import Dataset
from model.DataLoader import DataLoader
from model.analyzer import Analyzer as aly

# 设置参数
trainResultPath = ""
transformName = ""

# 加载分析器
alyer = aly.Analyzer(trainResultPath)
assert alyer.checkResult(), "训练结果不完整，不建议进行验证"
transform = transform.function.getTransform(transformName)
net = resnet1d34(num_class)
data_length = 4096
sampling_step = 2048
batch_size = 16  # 这里要设置成和训练一样，要不然会影响精度！！！
val_set_path = os.path.join(conf.getDatasetInfo("v4_wms")["paths"]["mac"], "val")  # dataset device

weight_path = os.path.join(train_result, "weight.pth")
if not os.path.isfile(weight_path):
    files = os.listdir(train_result)
    for file in files:
        if file.endswith(".pth"):
            weight_path = os.path.join(train_result, file)
            break

# 加载数据集
val_dataset = flowDataset(path=val_set_path, length=data_length, step=sampling_step, name="Validation Set")
val_loader = flowDataLoader(dataset=val_dataset, batch_size=batch_size, transform=transform, showInfo=True)

# 加载模型
device = torch.device(("cuda:0" if torch.cuda.is_available() else "cpu"))
print("Using device {}".format(device))
net = net.to(device)
net.load_state_dict(torch.load(weight_path, map_location=device))
loss_function = torch.nn.CrossEntropyLoss()

# 数据保存相关内容
model_name = net.__class__.__name__
task_name = f"[{model_name}]{xtime.getDateTimeForPath()}"
log_path = os.path.join(os.getcwd(), "result", "val", task_name)
info_path = os.path.join(log_path, "info.yaml")
iter_path = os.path.join(log_path, "iter")
result_path = os.path.join(log_path, "results.yaml")
model_param_amount = formatter.xNumFormat(sum([p.nelement() for p in net.parameters()]), unit="m", keep_float=2)
os.makedirs(log_path)

task_info = {
    "Task_Time": xtime.getDateTime(),
    "Model_Name": model_name,
    "Data_Length": data_length,
    "Sampling_Step": sampling_step,
    "Loss_Function": loss_function.__class__.__name__,
    "Transform": transform.__class__.__name__,
    "Model_Parameters_Amount": model_param_amount,
    "Batch_Size": batch_size,
    "Val_Set_Info": val_dataset.getDatasetInfoDict(),
}

confusion_matrix = [[0 for i in range(num_class)] for j in range(num_class)]

task_message = \
    f"""
    Start validation task!
    Model Name:{model_name}
    Data Length: {data_length}\tSampling Step: {sampling_step}
    Network Name: {model_name}\tTransform:{transform.__class__.__name__}
    Batch Size: {batch_size}\tModel_Parameters_Amount: {model_param_amount}
    """
printer.xprint_cyan(task_message)


def main():
    # 准备工作
    print("Begin Validation Process!")

    with open(info_path, "w") as info_fp:
        yaml.dump(task_info, info_fp)
    with open(iter_path, "w") as iter_fp:
        iter_fp.write(f"{'No':>6}\t{'Label':>8}\t{'Predict':>8}\t'Path'\n")
    with torch.no_grad():
        acc = 0
        count = 0
        while val_loader.isReadable():
            data, label, path = val_loader.getData()
            data = data.to(device)
            label = torch.Tensor(label).to(device)
            predict_y = net(data)
            predict_label = torch.argmax(predict_y, dim=1)
            for i in range(len(predict_label)):
                prl, tl = int(predict_label[i]), int(label[i])
                if prl == tl:
                    acc += 1
                count += 1
                with open(iter_path, "a+") as iter_fp:
                    iter_fp.write(f"{count:>6}\t{tl:>8}\t{prl:>8}\t{path[i].split('/')[-1]}\n")
                confusion_matrix[prl][tl] += 1
        normal_confusion_matrix = [[round(col / sum(line), 2) for col in line] for line in confusion_matrix]

        result_info = {
            "Sample_Amount": count,
            "Accurate_Amount": acc,
            "Accuracy": formatter.xNumFormat(acc / count, unit='%', keep_float=2),
            "Confusion_Matrix": confusion_matrix,
            "Normalized_Matrix": normal_confusion_matrix
        }
        with open(result_path, 'w') as result_fp:
            yaml.dump(result_info, result_fp)
        # val_loader.getData()
        print(f"Accuracy:{formatter.xNumFormat(acc / count, unit='%', keep_float=2)}")


if __name__ == "__main__":
    main()
