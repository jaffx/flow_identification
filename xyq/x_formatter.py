import math


def intFormatter(value, unit='k', keep_float: int = 2):
    if unit == 'k':
        value = value / 1000
    elif unit == 'm':
        value = value / 1e6
    value = round(value, keep_float)
    return f"{value}{unit}"

