# -*- coding:utf-8 -*-
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import os
from xyq import x_time as xtime
from xyq import x_printer as printer
from xyq import x_formatter as formatter
from model.model_v2 import MobileNetV2
from DataLoader.Dataset import flowDataset
from DataLoader.DataLoader import flowDataLoader
from DataLoader.transforms import toTensor, flowHilbertTransform


def main():
    # 定义训练设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 自定义训练参数
    data_length = 128 * 128
    sampling_step = 128 * 64
    batch_size = 8
    epoch_num = 100
    train_set_path, train_set_name = "../FlowDataset/Datas2/train", "TrainSet"
    val_set_path, val_set_name = "../FlowDataset/Datas2/val", "ValSet"
    learn_rate = 0.0001

    # 导入训练数据
    train_set = flowDataset(path=train_set_path, length=data_length, step=sampling_step, name=train_set_name)
    train_set_info = train_set.getDatasetInfo()
    val_set = flowDataset(path=val_set_path, length=data_length, step=sampling_step, name=val_set_name)
    val_set_info = val_set.getDatasetInfo()
    transform = flowHilbertTransform(7)
    train_loader = flowDataLoader(dataset=train_set, batch_size=batch_size, transform=transform, showInfo=True)
    val_loader = flowDataLoader(dataset=val_set, batch_size=batch_size, transform=transform, showInfo=True)

    # 定义模型
    net = MobileNetV2(4)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=learn_rate)
    model_param_amount = sum([p.nelement() for p in net.parameters()])

    # 设置数据记录参数
    model_name = net.__class__.__name__
    date_time_path = xtime.getDateTimeForPath()
    date_time = xtime.getDateTime()
    task_name = f"{date_time_path} [{model_name}]"
    log_path = os.path.join(os.getcwd(), 'logs', 'train', task_name)
    os.makedirs(log_path)
    weight_path = os.path.join(log_path, f'{task_name}.pth')
    info_fp_path = os.path.join(log_path, 'info')
    train_iter_fp_path = os.path.join(log_path, 'train_iter')
    val_iter_fp_path = os.path.join(log_path, 'val_iter')
    epoch_fp_path = os.path.join(log_path, 'epoch')
    console_log_file = os.path.join(log_path, 'console_log')
    printer.TARGET_FILES = console_log_file

    # 记录模型数据
    with open(info_fp_path, 'a+') as info_fp:
        info_fp.write(f"# TASK_TIME: {date_time}\n")
        info_fp.write(f"# Train Set Info\n")
        for line in train_set_info:
            info_fp.write('\t' + line + '\n')
        info_fp.write(f"# Val Set Info\n")
        for line in val_set_info:
            info_fp.write('\t' + line + '\n')
        info_fp.write(f"# Transform: {transform.__class__.__name__}\n")
        info_fp.write(f"# Data Length: {data_length} \t Sampling Step:{sampling_step}\n")
        info_fp.write(f"# Epoch Num: {epoch_num} \t Batch Size: {batch_size}\n")
        info_fp.write(
            f"# Model Name:{model_name} \t Model Parameters Amount:{formatter.intFormatter(model_param_amount, unit='m', keep_float=3)} \n")
        info_fp.write(f"# Optimizer:{optimizer.__class__.__name__} \t Learn Rate: {learn_rate}\n")
        info_fp.close()

    # 开始训练
    best_acc = 0
    with open(epoch_fp_path, 'a+') as epoch_fp:
        epoch_fp.write(f" {'Epoch':5}\t{'Train Time':10}\t{'Val Time':10}\t"
                       f"{'Train Loss':8}\t{'Train Batch':8}\t{'TNSample':8}\t{'Train NAcc':8}\t{'Train ACC':8}\t"
                       f"{'Val Loss':8}\t{'Val Batch':8}\t{'VNSample':8}\t{'Val NAcc':8}\t{'Val ACC':8}\n"
                       )
        epoch_fp.close()
    with open(train_iter_fp_path, 'a+') as titer_fp:
        titer_fp.write(
            f"{'Method':>6}\t{'epoch':>6}\t{'batch':>6}\t{'NSample':>6}\t{'AccNum':>6}\t{'ACC':>6}\t{'Loss':>6}\t{'AVGLoss':>6}\t{'Predict Label':32}\t{'Truth Label':32}\n")
        titer_fp.close()
    with open(val_iter_fp_path, 'a+') as viter_fp:
        viter_fp.write(
            f"{'Method':>6}\t{'epoch':>6}\t{'batch':>6}\t{'NSample':>6}\t{'AccNum':>6}\t{'ACC':>6}\t{'Loss':>6}\t{'AVGLoss':>6}\t{'Predict Label':32}\t{'Truth Label':32}\n")
        viter_fp.close()
    for epoch in range(epoch_num):
        # Epoch初始化
        train_batch_num, val_batch_num = 0, 0
        train_acc_num, val_acc_num = 0, 0
        train_sample_num, val_sample_num = 0, 0
        train_total_loss, val_total_loss = 0, 0

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
            label = torch.tensor(label, dtype=torch.long).to(device)
            # 正向传播
            predict_y = net(data)
            train_loss = loss_function(predict_y, label)
            # 反向传播
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # 训练数据统计
            batch_loss = train_loss.item()
            predict_label = torch.argmax(predict_y, dim=1)
            acc_num = torch.eq(predict_label, label).sum().item()
            sample_num = len(label)
            predict_label = predict_label.tolist()
            label = label.tolist()

            batch_acc = acc_num / sample_num
            sample_loss = batch_loss / sample_num

            train_sample_num += sample_num
            train_total_loss += batch_loss
            train_batch_num += 1
            train_acc_num += acc_num

            # 训练数据记录
            with open(train_iter_fp_path, 'a+') as titer_fp:
                titer_fp.write(
                    f"{'T':>6}\t{epoch:>6}\t{train_batch_num:>6}\t{sample_num:>6}\t{acc_num:>6}\t{batch_acc:>6.04}\t{batch_loss:>6.03}\t{sample_loss:>6.03}\t{str(predict_label)}\t{str(label)}\n")
                titer_fp.close()
        # 训练收尾
        train_loader.getData()
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
            label = torch.tensor(label, dtype=torch.long).to(device)
            # 正向传播
            predict_y = net(data)
            val_loss = loss_function(predict_y, label)

            # 测试数据统计
            batch_loss = val_loss.item()
            predict_label = torch.argmax(predict_y, dim=1)
            acc_num = torch.eq(predict_label, label).sum().item()
            sample_num = len(label)
            predict_label = predict_label.tolist()
            label = label.tolist()

            batch_acc = acc_num / sample_num
            sample_loss = batch_loss / sample_num

            val_sample_num += sample_num
            val_total_loss += batch_loss
            val_batch_num += 1
            val_acc_num += acc_num

            # 测试数据记录
            with open(val_iter_fp_path, 'a+') as viter_fp:
                viter_fp.write(
                    f"{'V':>6}\t{epoch:>6}\t{val_sample_num:>6}\t{sample_num:>6}\t{acc_num:>6}\t{batch_acc:>6.4}\t{batch_loss:>6.3}\t{sample_loss:>6.3}\t{str(predict_label)}\t{str(label)}\n")
                viter_fp.close()

        # 测试收尾
        val_loader.getData()
        val_end_time = time.time()
        # Epoch数据统计
        train_acc = train_acc_num / train_sample_num if train_sample_num else 0
        val_acc = val_acc_num / val_sample_num if val_sample_num else 0
        train_avg_loss = train_total_loss / train_sample_num if train_sample_num else 0
        val_avg_loss = val_total_loss / val_sample_num if val_sample_num else 0
        train_running_time = xtime.secsToStr(int(train_end_time - train_start_time))
        val_running_time = xtime.secsToStr(int(val_end_time - train_end_time))
        line_sign = ' '

        # 权重保存
        if best_acc < val_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), weight_path)
            line_sign = '*'
        # 数据保存
        printer.xprint_cyan(
            f"{line_sign}Epoch {epoch} \t Train ACC:{train_acc * 100:.2f}% \t Val ACC:{val_acc * 100:.2f}% \t Train Loss{train_avg_loss:.2f}\tVal Loss{val_avg_loss:.2f}"
        )
        with open(epoch_fp_path, 'a+') as epoch_fp:
            epoch_fp.write(f"{line_sign}{epoch:>5d}\t{train_running_time:>10}\t{val_running_time:10}\t"
                           f"{train_avg_loss:>8.3f}\t{train_batch_num:>8}\t{train_sample_num:>8}\t{train_acc_num:>8}\t{train_acc:>8.3f}\t"
                           f"{val_avg_loss:>8.3f}\t{val_batch_num:>8}\t{val_sample_num:>8}\t{val_acc_num:>8}\t{val_acc:>8.3f}\n"
                           )
            epoch_fp.close()


if __name__ == '__main__':
    main()
