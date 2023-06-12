#!/bin/bash

if [ ! -d "/hy-tmp" ]; then
  echo "\033[31m不是恒源云环境\033[0m"
  exit 1
fi

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
unzip v1_WMS_Simple_B.zip &
oss cp oss://datasets/v2_WMS_Sim_B.zip  .
unzip v2_WMS_Sim_B.zip

exit 0