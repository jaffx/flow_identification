# Flow_Identification 流型识别
*@author 徐永麒*</br>
*@email xuyongqi.lyn@qq.com*

本人的硕士毕业课题项目代码 </br>
如有需要，联系本人 </br>


# 〇、基础篇

## 下载代码
```shell
# 全量下载代码
git clone https://github.com/jaffx/flow_identification.git


# 更新代码
git pull origin main
```

## 启动训练

```shell
# 指定数据集
python script/train/train.py -d v1_wms

# 指定预处理方案
python run/train.py -d v3_wms -t aug0.2

# 指定数据长度和步长
python run/train.py -d v3_wms -l 4096 -s 2048

# 指定数据epoch和batchsize
python run/train.py -d v3_wms -e 100 -b 2048
```

## 代码仓库

[仓库地址点这里](https://github.com/jaffx/flow_identification)

```shell
# ssh方式
git clone git@github.com:jaffx/flow_identification.git
#http方式
git clone https://github.com/jaffx/flow_identification.git
```

## 提交代码

```shell
# 这是xcmd.sh中添加的快捷add指令
sh xcmd.sh add
```

# 数据集

## 数据集格式

### 概念

#### 1. 数据集格式

数据集格式主要指数据文件的存放格式，即**数据集文件树的模式**。

- **A格式**

```
 # 未进行数据集分割的格式，但已经已进行了数据标注
 # 格式为：数据集-子类-样本文件
```

- **B格式**

``` text
  # 四级目录 数据集-训练/测试集-子类-样本文件
  # 进行了分割，包括训练集和测试集
  WMS_Simple_B---# 数据集
  ├── train------# 训练集目录
  │    ├── 0-----# 训练集子类
  │    ├── 1
  │    ├── 2
  │    └── 3
  └── val-------# 测试集目录
      ├── 0-----# 测试集子类
      ├── 1
      ├── 2
      └── 3
```

- A格式 -> B格式 数据集分割命令

```shell
  python script/Dataset_Divide.py <A格式路径> <B格式路径>
```

#### 2. 数据组织格式

数据组织格式指数据文件内部的数据存放格式，与数据文件的解析有关。

- **Source**格式
  原始数据格式，每行数据有多个字段
- **Simple**格式 【推荐】

```text
* 简单数据格式
* 【重点】每行数据只有一个字段
* 【重点】没有表头，可以直接读取
* 【重点】数据集应当处理成这种格式进行训练
```

## 一期数据 WMS

1. 一期数据只有WMS数据，数据量较少。
2. 数据进行了标注，序号含义如下

| 序号  | 流型    |
|-----|-------|
| 0   | 段塞流   |
| 1   | 伪段塞流  |
| 2   | 分层波浪流 |
| 3   | 分层平滑流 |

## 二期数据 WMS+Pressure

二期数据量较大，且未进行标注。其中压力信号需要进行预处理。需要依赖一期训练出的模型进行标注。

### 处理脚本

    script/new_data_process

# 三、使用篇

本工程基于pytorch框架搭建，自行开发了一些适合流型数据处理的工具。使用方法与pytorch中提供的工具有相似之处，但
具体细节还以实物为准。

# 工具篇

该部分主要介绍自行开发的处理工具

## DataLoader 数据加载器

### 1.1 flowData

#### 基本功能

1. 保存一条流型数据，也就是一个数据文件的内容
2. 获取指定长度（length）的数据段，并将**数据读取位置** *（数据指针）* 向后移动一定步长（step）。
   *（【注意】只实现"取+移动"的功能，并不决定数据长度和步长）*
3. 判断该条flowData中是否还能取出长度为*length*的数据
4. 保存该数据的信息，包括文件名、标签等


### 1.2 flowDataset

#### 基本功能

1. 保存一个数据集的数据+信息 *（训练集or测试集中的一个）*
2. 可以获取一个batch的数据 *（不决定batch的size）*
3. 保存若干个flowData，保存需要从flowData中读取的数据长度和移动步长
4. 判断一个数据集是否还有数据可读 *（判断是否还有可以读取的flowData）*



### 1.3 flowDataLoader

#### 基本功能

1. 用来训练or测试的数据结构
2. 保存一轮训练的信息，如训练时长，batch的个数，预估剩余时长等
3. 内部保存一个flowDataset
4. 保存BatchSize,每次从flowDataset中读取BatchSize大小的数据段。


## Transform 转换器

pytorch提供了transform模块，但使用起来不顺手，因此重新设计该模块。

### 基本功能

1. 数据预处理（标准化、傅里叶变换 等数学处理）
2. 数据格式处理 （升维/降维，tensor转化）
3. 数据增强 （随机噪声、随机失活等）

### 使用

#### 预定义transform

1. 在xlib/utils/transform.__init__.py中添加定义

```python
from xlib.transforms import BaseTrans as BT  # 基本模块
from xlib.transforms import Preprocess as PP  # 预处理模块
from xlib.utils.transform.transform import addTransform

# 1. 定义transform
normalization = BT.transform_set([
    PP.normalization(),  # 先标准化
    BT.toTensor()  # 标准化的结果转化为tensor类型
])
# 2.注册。
# addTransform("名字",transform,"描述")
addTransform("normalization", normalization, "标准化")
```

#### 使用预定义的transform

程序中使用预定义transform

```python
from xlib.conf import conf

train_transform = conf.getTransform("normalization")
```

训练时指定transform

```shell
# -t参数用来指定训练集transform
python run/train.py -d test -t "aug0.2"
```

#### transform基本原理

```python
class transform_base:
    def __init__(self):
        pass

    def __call__(self, x):
        # 输入一组数据，返回对数据的处理结果
        return x

    def __str__(self):
        # 获取该转换器的字符描述
        return f"{self.__class__.__name__}()"
```

#### 使用transform进行数据增强

```python
# 基本模块
from xlib.transforms import BaseTrans as BT  
# 预处理模块
from xlib.transforms import Preprocess as PP  
# 数据增强模块
from xlib.transforms import DataAugmentation as DA  
from xlib.utils.transform.transform import addTransform

aug1 = BT.transform_set([
    # 顺序触发下列transform
    PP.normalization(),
    BT.random_trigger( 
        # 0.2概率触发内部transform，不触发就是恒等变换
        BT.random_selector([
            # 随机从下列transform中选择一个触发
            DA.random_range_masking(0.1),
            DA.random_noise(-0.05, 0.05),
            DA.normalized_random_noise(0, 0.05),
        ]),
        prob=0.2
    ),
    BT.toTensor()
])
addTransform("aug0.2", aug1, "概率为0.2的小强度数据增强")
```

## Modifier 修改器





