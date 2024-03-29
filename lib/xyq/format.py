import datetime
import time

from . import printer


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


def showRuningTime(func):
    def getTimeMsInner(*args, **kwargs):
        time0 = time.time()
        func(*args, **kwargs)
        runtime = int((time.time() - time0) * 1000)
        if runtime < 1:
            runtime = f"{runtime}ms"
        else:
            runtime = f"{runtime / 1000}s"
        printer.xprint(f"Function<{func.__name__}{args}{kwargs}>running time:{runtime}")

    return getTimeMsInner


def getDateTime():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def getDateTimeForPath():
    return datetime.datetime.now().strftime('%Y%m%d.%H%M%S')


def secsToStr(secs: int):
    assert isinstance(secs, int), "secs must be an integer!"
    second = secs % 60
    secs //= 60
    minute = secs % 60
    secs //= 60
    hour = secs

    return f"{hour:02}:{minute:02}:{second:02}"
