#!/bin/bash

# 开始训练之前必须登录oss
oss ls -s
if [ $? -ne 0 ]; then
  echo "\033[31m oss未登录 \033[0m"
  exit 1
fi

# 实验备注，详细填写
comment="探究特征复用方式和失活比例效果实验"
# 训练集transform
vt="ms-normalization"
# 迭代论数
epoch="80"
# 数据集名称
dataset="mv1"
# 神经网络结构
net="MSFINet-ConvFusion"
Nets=("MSFINet-ConvFusion" "MSFINet-FullTrunk" "MSFINet-HalfBranch")
TTs=("ms-invalidator-20%" "ms-invalidator-30%" "ms-invalidator-40%" "ms-invalidator-50%")
for net in ${Nets[*]}; do
  for tt in ${TTs[*]}; do
    python script/train/mhnet_train.py --net "$net" --dataset "$dataset" --tt="$tt" --vt="$vt" --epochs "$epoch" --comment "$comment" >>out.txt
  done
done

sh xcmd.sh move_result
shutdown
