#!/bin/bash

if [ ! -d "/hy-tmp" ]; then
  echo "\033[31m不是恒源云环境\033[0m"
  exit 1
fi

# 下载数据集
oss ls
if [ $? -ne 0 ]; then
  echo "\033[31m oss未登录 \033[0m"
  exit 1
fi
if [ ! -d /home/dataset ]; then
  mkdir -p /home/dataset && cd /home/dataset
else
  cd /home/dataset
fi
oss cp oss://datasets/v1_WMS_Simple_B.zip  .
unzip -q v1_WMS_Simple_B.zip &
oss cp oss://datasets/v3_WMS_Simple_B.zip .
unzip -q v3_WMS_Simple_B.zip &
oss cp oss://datasets/v4_WMS_Simple_B.zip .
unzip -q v4_WMS_Simple_B.zip &
oss cp oss://datasets/v4_Pressure_IDX3_Simple_B.zip .
unzip -q v4_Pressure_IDX3_Simple_B.zip &
oss cp oss://datasets/v4_Pressure_IDX4_Simple_B.zip  .
unzip -q v4_Pressure_IDX4_Simple_B.zip &
oss cp oss://datasets/v4_DiffPressure_IDX3-4_Simple_B.zip .
unzip -q v4_DiffPressure_IDX3-4_Simple_B.zip &
oss cp oss://datasets/mv1.zip .
unzip -q mv1.zip

# python环境
# cd ~/flow_identification && pip install -r requirements.txt

# 更新代码
git pull origin main

# git 设置
git config --global user.name HY
git config --global user.email HY@xyq.com

exit 0