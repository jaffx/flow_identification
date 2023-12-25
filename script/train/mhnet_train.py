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
from model.net.MHNet import MHNet
from model.Dataset.MSDataset import MSDataset
from model.DataLoader.DataLoader import flowDataLoader
from lib import xyq


def dealArgs():
    parser = argparse.ArgumentParser(description='train')
    # 添加命令行参数
    parser.add_argument('-d', '--dataset', type=str, required=True,
                        help='数据集名称，定义在：conf/dataset/dataset_path.yaml.')
    parser.add_argument('-e', '--epochs', type=int, default=80, help='训练epoch数量，默认50')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='BatchSize, 默认64')
    parser.add_argument('-l', '--length', type=int, default=4096, help='数据长度')
    parser.add_argument('-s', '--step', type=int, default=2048, help='数据采样步长')
    parser.add_argument('--tt', type=str, default="ms-normalization", help='指定训练集transform')
    parser.add_argument('--vt', type=str, default="ms-normalization", help='指定测试集transform')
    parser.add_argument('-c', '--comment', type=str, required=True, help="训练实验备注，详细填写")
    parser.add_argument('--lr', type=float, default=0.00001, help='训练初始学习率')
    parser.add_argument('--mod', type=str, default="None", help='训练epoch修改器')
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
    dataset_path, class_num = xyq.function.getMSDatasetInfo(dataset=dataset_name, device=device_name)
    train_set_path, train_set_name = os.path.join(dataset_path, "train"), f"{dataset_name}-train"
    val_set_path, val_set_name = os.path.join(dataset_path, "val"), f"{dataset_name}-val"
    # 加载transform
    trainTransformName, valTransformName = args.tt, args.vt
    train_transform = xyq.function.getTransform(trainTransformName)
    val_transform = xyq.function.getTransform(valTransformName)

    modifier = xyq.function.getModifier(args.mod)
    xyq.printer.xprint("开始训练,信息如下")
    xyq.printer.xprint(f"\tepoch {epoch_num},\tbatch_size {batch_size},\t lr {args.lr}\n"
                       f"\tdataset {dataset_name},\tlength {data_length},\tstep {sampling_step}\n"
                       f"\ttrain_transform: {trainTransformName}"
                       )
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        xyq.printer.xprint(f"using device : {device}")
    else:
        device = torch.device("cpu")
        xyq.printer.xprint_red(f"using device : {device}")
    # 导入训练数据
    train_set = MSDataset(path=train_set_path, length=data_length, step=sampling_step, name=train_set_name)
    train_set.getDatasetInfo()
    val_set = MSDataset(path=val_set_path, length=data_length, step=sampling_step, name=val_set_name)
    val_set.getDatasetInfo()

    train_loader = flowDataLoader(dataset=train_set, batch_size=batch_size, transform=train_transform, showInfo=True)
    val_loader = flowDataLoader(dataset=val_set, batch_size=batch_size, transform=val_transform, showInfo=True)

    # 定义+初始化 模型&优化器
    net = MHNet(class_num)
    net = net.to(device)
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=args.lr)
    model_param_amount = sum([p.nelement() for p in net.parameters()])
    # 损失函数
    loss_function = nn.CrossEntropyLoss()

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
        "Task_Name": "Model_training_MultiHeadNet",
        "Task_Time": date_time,
        "Dataset": dataset_name,
        "Device_Name": device_name,
        "Class_Num": class_num,
        "Train Transform": trainTransformName,
        "Validation Transform": valTransformName,
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
        "Modifier": args.mod,
        "Comment": args.comment,
    }
    yaml.dump(task_info, open(info_fp_path, "w", encoding='utf-8'), allow_unicode=True)

    # 开始训练
    best_acc = 0
    learn_rate = args.lr

    with open(epoch_fp_path, 'a+') as epoch_fp:
        epoch_fp.write(f" {'Epoch':5}\t{'Train Time':10}\t{'Val Time':10}\t{'LR':8}\t{'Trans':10}\t"
                       f"{'TLoss':8}\t{'T Batch':8}\t{'TNSample':8}\t{'TNAcc':8}\t{'TACC':8}\t"
                       f"{'VLoss':8}\t{'V Batch':8}\t{'VNSample':8}\t{'VNAcc':8}\t{'VACC':8}\n"
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
        train_acc_num, val_acc_num = 0, 0
        train_sample_num, val_sample_num = 0, 0
        train_total_loss, val_total_loss = 0, 0

        train_start_time = time.time()
        # 训练过程
        # 训练初始化
        net.train()
        train_loader.Init()
        # 开始训练
        while train_loader.isReadable():
            # 读取数据

            data, label, path = train_loader.getData()
            for i in range(len(data)):
                if data[i] is not None:
                    data[i] = data[i].to(device)
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
                    f"{'T':>6}\t{epoch:>6}\t{train_batch_num:>6}\t{sample_num:>6}\t{acc_num:>6}\t{batch_acc:>6.04}\t{batch_loss:>6.03}\t{sample_loss:>6.03}\n")
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

        while val_loader.isReadable():
            data, label, path = val_loader.getData()
            for i in range(len(data)):
                if data[i] is not None:
                    data[i] = data[i].to(device)
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
                    f"{'V':>6}\t{epoch:>6}\t{val_batch_num:>6}\t{sample_num:>6}\t{acc_num:>6}\t{batch_acc:>6.4}\t{batch_loss:>6.3}\t{sample_loss:>6.3}\n")
                viter_fp.close()

        # 测试收尾
        val_loader.getData()  # 输出一下dataloader一轮结束后的信息
        val_end_time = time.time()
        # Epoch数据统计
        train_acc = train_acc_num / train_sample_num if train_sample_num else 0
        val_acc = val_acc_num / val_sample_num if val_sample_num else 0
        train_avg_loss = train_total_loss / train_sample_num if train_sample_num else 0
        val_avg_loss = val_total_loss / val_sample_num if val_sample_num else 0
        train_running_time = xyq.format.secsToStr(int(train_end_time - train_start_time))
        val_running_time = xyq.format.secsToStr(int(val_end_time - train_end_time))
        line_sign = ' '  # epoch结果中的行首标志，最优为*

        # 权重保存
        if best_acc < val_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), weight_path)
            line_sign = '*'
        # 数据保存
        printer.xprint_cyan(
            "    ".join([
                f"{line_sign}Epoch {epoch}",
                f"Train ACC:{train_acc * 100:.2f}%",
                f"Val ACC:{val_acc * 100:.2f}% ",
                f"Train Loss{train_avg_loss:.2f}",
                f"Val Loss{val_avg_loss:.2f}"
            ]
            )
        )
        # epoch数据保存
        with open(epoch_fp_path, 'a+') as epoch_fp:
            epoch_fp.write(
                f"{line_sign}{epoch:>5d}\t{train_running_time:>10}\t{val_running_time:>10}\t{learn_rate:>8}\t"
                f"{trainTransformName:>10}\t"
                f"{train_avg_loss:>8.6f}\t{train_batch_num:>8}\t{train_sample_num:>8}\t{train_acc_num:>8}\t{train_acc:>8.4f}\t"
                f"{val_avg_loss:>8.6f}\t{val_batch_num:>8}\t{val_sample_num:>8}\t{val_acc_num:>8}\t{val_acc:>8.4f}\n"
            )
            epoch_fp.close()


if __name__ == '__main__':
    main()
