import json

from model.config import config
import re

"""
:brief ModifierEpoch 训练修改器，根据Epoch修改训练参数
配置文件路径 
    xyq/epoch_modifier.json
配置文件格式
    {
        "modifier_1_name"：{
            "desc":"修改器的描述",
            "mod"{
                "20":{
                    "lr":{
                        "action" : "set",
                        "value" : 1e-6,
                        "desc": "在epoch=20的时候把学习率设置为1e-6"
                    },
                    "transform"{
                        ... 对transform的修改内容
                    }
                },
                "condition_2":{
                    ... 条件2的内容
                }
            }
        }
    }
condition规则
    示例     规则          描述               优先级
    *       all规则       任意epoch都触发      1
    20,30   区间规则       在20到30epoch触发    2
    20,-    半开区间       20到结束触发         2
    20      精准epoch     在第20个epoch触发    3
"""


class ModifierEpoch:
    """
    :brief
        Epoch维度的修改器，根据配置对训练参数进行调整
    :设计原则
        1. 使用前严格，有错必报
        2. 使用时宽容，尽量容忍
        3. 所有问题在训练前报出，尽量避免中断训练
    """
    range_pattern = re.compile(r"(?P<start>\d+), *(?P<end>\d+)")
    half_range_pattern = re.compile("(?P<start>\d+), *-")
    num_pattern = re.compile("\d+")

    @staticmethod
    def __getModifierByName(name):
        """
        获取指定的修改器
        :param name 修改器名称
        :return dict|None
        """
        __modconf_path = "xyq/epoch_modifier.json"
        mod_config = config.config(__modconf_path)
        conf = mod_config.get(name)
        if conf is None:
            conf = mod_config.get()
            print(f"【{name}】modifier 不存在，按照如下方式设置")
            for key in conf:
                desc = mod_config.get(key + "/desc")
                desc = "no description" if desc is None else desc
                print(f"\t{key:<8}\t{desc:<8}")
            raise Exception("Modifier not exist")
        return conf

    @staticmethod
    def checkConf(conf: dict):
        """
        :param conf 配置
        """
        if "desc" not in conf or "mod" not in conf:
            return False
        return True

    def __init__(self, name: str, total_epoch=50):
        mod_conf = ModifierEpoch.__getModifierByName(name)
        assert self.checkConf(mod_conf), "mod配置错误"
        self.mods = {}
        self.total_epoch = total_epoch
        self.desc = mod_conf["desc"]
        self.parseConf(mod_conf["mod"])

    def add1Modification(self, epoch: int, modset: dict, nice=0):
        """
        增加一个mod
        :param epoch 在哪个epoch触发
        :param modset 触发的mod集合
        :param nice 本次mod的优先级
        """
        if epoch < 0 or epoch >= self.total_epoch:
            return False
        if epoch not in self.mods:
            self.mods[epoch] = {}

        for mod_name in modset:
            if mod_name not in self.mods[epoch]:
                self.mods[epoch][mod_name] = {
                    "mod_name": mod_name,
                    "nice": nice,
                    "mod_detail": modset[mod_name]
                }
            else:
                pre_nice = self.mods[epoch][mod_name]["nice"]
                if pre_nice > nice:
                    continue
                self.mods[epoch][mod_name] = {
                    "mod_name": mod_name,
                    "nice": nice,
                    "mod_detail": modset[mod_name]
                }
        return True

    def parseConf(self, mod_conf):
        print(f"开始解析Modifier...")
        for condition in mod_conf:
            modset = mod_conf[condition]
            condition = condition.strip()
            if condition == "*":
                nice = 1
                for epoch in range(self.total_epoch):
                    self.add1Modification(epoch, modset, nice)
            elif self.range_pattern.match(condition):
                nice = 2
                ret = self.range_pattern.match(condition)
                start = int(ret.group("start"))
                end = int(ret.group("end"))
                if start > end:
                    continue
                for epoch in range(start, end + 1):
                    self.add1Modification(epoch, modset, nice)
            elif self.half_range_pattern.match(condition):
                nice = 2
                ret = self.half_range_pattern.match(condition)
                start = int(ret.group("start"))
                end = self.total_epoch
                if start > end:
                    continue
                for epoch in range(start, end + 1):
                    self.add1Modification(epoch, modset, nice)
            elif self.num_pattern.match(condition):
                nice = 3
                epoch = int(condition)
                self.add1Modification(epoch, modset, nice)
            else:
                print(f"非法Condition【{condition}】")
                continue
        print(f"Modifier解析完成")

    def getModStr(self, action="return"):
        if action == "show":
            print(self.mods)
        elif action == "show_json":
            print(json.dumps(self.mods))
        return json.dumps(self.mods)

    def mod_or_not(self, epoch: int, mod_type=None):
        if epoch not in self.mods:
            return False

        if mod_type is None:
            return True

        if mod_type not in self.mods[epoch]:
            return False
        return True

    def mod_lr(self, epoch: int, lr: float):
        if not self.mod_or_not(epoch, "lr"):
            return lr
        # 对lr进行调整，返回新的lr
        lr_mod = self.mods[epoch]["lr"]["mod_detail"]
        if "action" not in lr_mod or "value" not in lr_mod:
            return lr
        action = lr_mod["action"]
        value = lr_mod["value"]

        try:
            if action in ("set", "="):
                lr = value
            elif action in ("divide", "/"):
                lr /= value
            elif action in ("multiply", "*"):
                lr *= value
            else:
                lr = lr
        except Exception as e:
            lr = lr
        print(f"Modifier@{epoch}  lr 修改, action={action}, value = {value}, lr = {lr}")
        return lr
