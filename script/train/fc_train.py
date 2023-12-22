# -*- coding:utf-8 -*-
import os
import sys
import time
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(".")
from lib.xyq import printer as printer, format as formatter
from model.net.FC.FCRes1D import FCRes1D
from model.Dataset.FCDataset import FCDataset
from model.DataLoader.DataLoader import flowDataLoader
from lib import xyq


def dealArgs():
    parser = argparse.ArgumentParser(description='train')
    # 添加命令行参数
    parser.add_argument('-d', '--dataset', type=str, default="fc1", help='训练使用的数据集，见conf/dataset/info.yaml')
    parser.add_argument('-e', '--epochs', type=int, default=50, help='Number of epochs to train.')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='Number of batch size to train.')
    parser.add_argument('-l', '--length', type=int, default=4096, help='Data Length')
    parser.add_argument('-s', '--step', type=int, default=2048, help='Step Length')
    parser.add_argument('-t', '--transform', type=str, default="normalization", help='Transform for train method')
    parser.add_argument('--lr', type=float, default=0.00001, help='learn rate')
    parser.add_argument('--mod', type=str, default="None", help='epoch modifier')
    # 从命令行中结构化解析参数
    args = parser.parse_args()
    return args


def main():
    # 训练参数初始化
    args = dealArgs()

    data_length = args.length
    sampling_step = args.step
    batch_size = args.batch_size
    epoch_num = args.epochs
    dataset_name = args.dataset

    # 获取执行环境 mac本地/恒源云
    device_name = xyq.function.getDeviceName()
    # 加载数据集
    dataset_path = xyq.function.getDatasetPath(dataset=dataset_name, device=device_name)
    train_set_path, train_set_name = os.path.join(dataset_path, "train"), f"{dataset_name}-train"
    val_set_path, val_set_name = os.path.join(dataset_path, "val"), f"{dataset_name}-val"
    # 加载transform
    transform_name = args.transform
    train_transform = xyq.function.getTransform(args.transform)
    val_transform = xyq.function.getTransform("normalization")

    modifier = xyq.function.getModifier(args.mod)
    xyq.printer.xprint("开始训练,信息如下")
    xyq.printer.xprint(f"\tepoch {epoch_num},\tbatch_size {batch_size},\t lr {args.lr}\n"
                       f"\tdataset {dataset_name},\tlength {data_length},\tstep {sampling_step}\n"
                       f"\ttrain_transform: {args.transform}"
                       )
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        xyq.printer.xprint(f"using device : {device}")
    else:
        device = torch.device("cpu")
        xyq.printer.xprint_red(f"using device : {device}")
    # 导入训练数据
    train_set = FCDataset(path=train_set_path, length=data_length, step=sampling_step, name=train_set_name)
    train_set.getDatasetInfo()
    val_set = FCDataset(path=val_set_path, length=data_length, step=sampling_step, name=val_set_name)
    val_set.getDatasetInfo()

    train_loader = flowDataLoader(dataset=train_set, batch_size=batch_size, transform=train_transform, showInfo=True)
    val_loader = flowDataLoader(dataset=val_set, batch_size=batch_size, transform=val_transform, showInfo=True)

    # 定义+初始化 模型&优化器
    net = FCRes1D()
    net = net.to(device)
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=args.lr)
    model_param_amount = sum([p.nelement() for p in net.parameters()])
    # 损失函数
    loss_function = nn.MSELoss()

    # 设置数据记录参数
    model_name = net.__class__.__name__
    date_time_path = xyq.format.getDateTimeForPath()
    date_time = xyq.format.getDateTime()
    task_name = f"{date_time_path}_{model_name}"
    result_path = os.path.join(os.getcwd(), 'result', 'train', task_name)
    os.makedirs(result_path)
    # 权重保存路径 result/train/<文件名>.pth
    weight_path = os.path.join(result_path, f'weight.pth')
    # 任务信息保存路径
    info_fp_path = os.path.join(result_path, 'info.yaml')
    train_iter_fp_path = os.path.join(result_path, 'train_iter')
    val_iter_fp_path = os.path.join(result_path, 'val_iter')
    epoch_fp_path = os.path.join(result_path, 'epoch')
    console_log_file = os.path.join(result_path, 'console_log')
    printer.TARGET_FILES = console_log_file
    xyq.printer.xprint(
        f"Model:{model_name} BatchSize: {batch_size} "
        f"DataLength:{data_length} Step:{sampling_step} "
        f"Model Parameters {formatter.xNumFormat(model_param_amount, unit='m', keep_float=3)}")

    # 通过yaml文件记录模型数据
    task_info = {
        "Task_Name": "Flow Calculation",
        "Task_Time": date_time,
        "Dataset": dataset_name,
        "Device_Name": device_name,
        "Transform": args.transform,
        "Model_Name": model_name,
        "Model_Parameter_Amount": formatter.xNumFormat(model_param_amount, 'm', 3),
        "Data_Length": data_length,
        "Sampling_Step": sampling_step,
        "Train_Transform": train_transform.str(),
        "Val_Transform": val_transform.str(),
        "Batch_Size": batch_size,
        "Epoch_Num": epoch_num,
        "Learn_Rate": args.lr,
        "Optimizer": optimizer.__class__.__name__,
        "Loss_Function": loss_function.__class__.__name__,
        "Train_Set_Info": train_set.getDatasetInfoDict(),
        "Val_Set_Info": val_set.getDatasetInfoDict(),
        "modifier": args.mod
    }
    yaml.dump(task_info, open(info_fp_path, "w"))

    # 开始训练
    minMSE = 1e9
    learn_rate = args.lr
    with open(epoch_fp_path, 'a+') as epoch_fp:
        epoch_fp.write("\t".join([f" {'Epoch':5}",
                                  f"{'Train Time':10}",
                                  f"{'Val Time':10}",
                                  f"{'LR':8}",
                                  f"{'Trans':10}"
                                  f"{'T Batch':8}",
                                  f"{'TNSample':8}",
                                  f"{'trainMSE':8}",
                                  f"{'VBatch':8}",
                                  f"{'VNSample':8}",
                                  f"{'VMSE':8}"])
                       )
        epoch_fp.close()

    with open(train_iter_fp_path, 'a+') as titer_fp:
        titer_fp.write(
            f"{'Method':>6}\t{'epoch':>6}\t{'batch':>6}\t{'NSample':>6}\t{'AccNum':>6}\t{'ACC':>6}\t{'Loss':>6}\t{'AVGLoss':>6}\n")
        titer_fp.close()

    with open(val_iter_fp_path, 'a+') as viter_fp:
        viter_fp.write(
            f"{'Method':>6}\t{'epoch':>6}\t{'batch':>6}\t{'NSample':>6}\t{'AccNum':>6}\t{'ACC':>6}\t{'Loss':>6}\t{'AVGLoss':>6}\n")
        viter_fp.close()

    for epoch in range(epoch_num):
        # modifier干涉训练
        if modifier and modifier.mod_or_not(epoch):
            learn_rate = modifier.mod_lr(epoch, learn_rate)
            optimizer.lr = learn_rate
        # Epoch初始化
        train_batch_num, val_batch_num = 0, 0
        trainMSE, valMSE = 0, 0
        train_sample_num, val_sample_num = 0, 0

        train_start_time = time.time()
        # 训练过程
        # 训练初始化
        net.train()
        train_loader.Init()
        # 开始训练
        while train_loader.getReadable():
            # 读取数据
            data, label, path = train_loader.getData()
            data = data.to(device)
            label = torch.tensor(label, dtype=torch.float).to(device)
            # 正向传播
            predict_y = net(data)
            train_loss = loss_function(predict_y, label)
            # 反向传播
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # 训练数据统计
            batch_loss = train_loss.item()
            sample_num = len(label)
            sample_loss = batch_loss / sample_num
            train_sample_num += sample_num
            trainMSE += batch_loss
            train_batch_num += 1

            # 训练数据记录
            with open(train_iter_fp_path, 'a+') as titer_fp:
                titer_fp.write(
                    f"{'T':>6}\t{epoch:>6}\t{train_batch_num:>6}\t{sample_num:>6}\t{sample_loss:>6.4}\t{batch_loss:>6.03}\t{sample_loss:>6.03}\n")
                titer_fp.close()
        # 训练收尾
        train_loader.getData()  # 输出一下结束内容
        train_end_time = time.time()

        # 测试过程
        # 测试初始化
        net.eval()
        val_set.Init()
        # printer.xprint("Epoch{} val start at {}".format(epoch, xtime.getDateTime()))
        # 开始测试

        while val_loader.getReadable():
            data, label, path = val_loader.getData()
            data = data.to(device)
            label = torch.tensor(label, dtype=torch.float).to(device)
            # 正向传播
            predict_y = net(data)
            val_loss = loss_function(predict_y, label)

            # 测试数据统计
            batch_loss = val_loss.item()
            sample_num = len(label)
            sample_loss = batch_loss / sample_num
            val_sample_num += sample_num
            valMSE += batch_loss
            val_batch_num += 1
            # 测试数据记录
            with open(val_iter_fp_path, 'a+') as viter_fp:
                viter_fp.write(
                    f"{'V':>6}\t{epoch:>6}\t{val_batch_num:>6}\t{sample_num:>6}\t{sample_loss:>6.4}\t{batch_loss:>6.3}\t{sample_loss:>6.3}\n")
                viter_fp.close()

        # 测试收尾
        val_loader.getData()  # 输出一下dataloader一轮结束后的信息
        val_end_time = time.time()
        # Epoch数据统计
        trainMSE /= train_sample_num
        valMSE /= val_sample_num
        train_running_time = xyq.format.secsToStr(int(train_end_time - train_start_time))
        val_running_time = xyq.format.secsToStr(int(val_end_time - train_end_time))
        line_sign = ' '  # epoch结果中的行首标志，最优为*

        # 权重保存
        if minMSE > valMSE:
            minMSE = valMSE
            torch.save(net.state_dict(), weight_path)
            line_sign = '*'
        # 数据保存
        printer.xprint_cyan(
            "    ".join([
                f"{line_sign}Epoch {epoch}",
                f"Train MSE:{trainMSE:.2f}",
                f"Val MSE:{valMSE:.2f} ",
            ]
            )
        )
        # epoch数据保存
        with open(epoch_fp_path, 'a+') as epoch_fp:
            epoch_fp.write(
                f"{line_sign}{epoch:>5d}\t"
                f"{train_running_time:>10}\t"
                f"{val_running_time:>10}\t"
                f"{learn_rate:>8}\t"
                f"{transform_name:>10}\t"
                f"{train_batch_num:>8}\t"
                f"{train_sample_num:>8}\t"
                f"{trainMSE}\t"
                f"{val_batch_num:>8}\t"
                f"{val_sample_num:>8}\t"
                f"{valMSE}\t"
            )
            epoch_fp.close()


if __name__ == '__main__':
    main()
