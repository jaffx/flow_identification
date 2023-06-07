# Flow_Identification 流型识别
*@author: 徐永麒*

# 一、Git篇
## 1.1 代码仓库
    git@github.com:jaffx/flow_identification.git
## 1.2 提交代码
```shell
# 这是xcmd.sh中添加的快捷add指令
sh xcmd.sh add
```


# 二、数据集
## 一期数据 WMS
1. 一期数据只有WMS数据，数据量较少。
2. 数据进行了标注，序号含义如下

| 序号  | 流型    |
|-----|-------|
| 1   | 段塞流   |
| 2   | 伪段塞流  |
| 3   | 分层波浪流 |
| 4   | 分层平滑流 |

## 二期数据 WMS+Pressure
  二期数据量较大，且未进行标注。其中压力信号需要进行预处理。需要依赖一期训练出的模型进行标注。
### 处理脚本
    script/new_data_process

# 三、使用篇
## 1. 数据读取
## 2. 数据预处理
## 3. 训练&测试Å






