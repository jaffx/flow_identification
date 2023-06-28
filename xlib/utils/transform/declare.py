"""
定义在这里
"""
from xlib.transforms import BaseTrans as BT
from xlib.transforms import Preprocess as PP
from xlib.transforms import DataAugmentation as DA

from xlib.utils.transform.transform import addTransform

normalization = BT.transform_set([
    PP.normalization(),
    BT.toTensor()
])
addTransform("normalization", normalization, "标准化")

aug1 = BT.transform_set([
    PP.normalization(),
    BT.random_trigger(
        BT.random_selector([
            DA.random_range_masking(0.1),
            DA.random_noise(-0.05, 0.05),
            DA.normalized_random_noise(0, 0.05),
        ]),
        prob=0.2
    ),
    BT.toTensor()
])
addTransform("aug0.2", aug1, "概率为0.2的小强度数据增强")
aug2 = BT.transform_set([
    PP.normalization(),
    BT.random_trigger(
        BT.random_selector([
            DA.random_range_masking(0.1),
            DA.random_noise(-0.05, 0.05),
            DA.normalized_random_noise(0, 0.05),
        ]),
        prob=0.3
    ),
    BT.toTensor()
])
addTransform("aug0.3", aug2, "概率为0.3的小强度数据增强")
aug3 = BT.transform_set([
    PP.normalization(),
    BT.random_trigger(
        BT.random_selector([
            DA.random_range_masking(0.1),
            DA.random_noise(-0.05, 0.05),
            DA.normalized_random_noise(0, 0.05),
        ]),
        prob=0.4
    ),
    BT.toTensor()
])
addTransform("aug0.4", aug3, "概率为0.4的小强度数据增强")
