#!/bin/bash

# python run/train.py -d v4_wms --lr 0.0001 --mod "lr*0.8" >>out.txt
python run/train.py -d v4_pre_3 >>out.txt
python run/train.py -d v4_pre_4 >>out.txt
python run/train.py -d v4_dp_34 >>out.txt

# 训练收尾
sh xcmd.sh move_result
shutdown
