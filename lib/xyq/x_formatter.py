import math


def xNumFormat(value, unit='k', keep_float: int = 2):
    """
    :unit 单位
    :keep_float 保留小数位数
    """
    if unit == 'k':
        value = value / 1000
    elif unit == 'm':
        value = value / 1e6
    elif unit == '%':
        value = value * 100
    value = round(value, keep_float)
    return f"{value}{unit}"

