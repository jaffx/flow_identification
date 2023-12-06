#!/bin/bash

# 开始训练之前必须登录oss
oss ls -s
if [ $? -ne 0 ]; then
  echo "\033[31m oss未登录 \033[0m"
  exit 1
fi

# python script/train/mhnet_train.py --dataset mv1 --transform='ms-normalization' --epochs 80 --comment ""
python script/train/mhnet_train.py --dataset mv1 --transform='ms-invalidator-normalization' --comment "尝试使用数据源失活，概率20%"
python script/train/mhnet_train.py --dataset mv1 --transform='ms-normalization' --comment "epoch设置为80，在更多的迭代次数之下能不能达到更高精度"


sh xcmd.sh move_result
shutdown