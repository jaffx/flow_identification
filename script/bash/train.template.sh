#!/bin/bash

# 开始训练之前必须登录oss
oss ls -s
if [ $? -ne 0 ]; then
  echo "\033[31m oss未登录 \033[0m"
  exit 1
fi

# 实验备注，详细填写
comment="探究融合方式的影响，使用残差融合"
# 训练集transform
tt="ms-normalization"
# 测试集transform
vt="ms-normalization"
# 迭代论数
epoch="80"
# 数据集名称
dataset="mv1"
# 神经网络结构
net="MSFINet-ResFusion"
python script/train/mhnet_train.py --net "$net" --dataset "$dataset" --tt="$tt" --vt="$vt" --epochs "$epoch" --comment "$comment" >>out.txt


# 实验备注，详细填写
comment="探究融合方式的影响，使用特征向量向量融合"
# 训练集transform
tt="ms-normalization"
# 测试集transform
vt="ms-normalization"
# 迭代论数
epoch="80"
# 数据集名称
dataset="mv1"
# 神经网络结构
net="MSFINet-BasicFusion"
python script/train/mhnet_train.py --net "$net" --dataset "$dataset" --tt="$tt" --vt="$vt" --epochs "$epoch" --comment "$comment" >>out.txt

sh xcmd.sh move_result
shutdown
