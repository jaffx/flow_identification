import datetime
import time
import xyq.x_printer as printer


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


def getTimeNow():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def secsToStr(secs: int):
    assert isinstance(secs, int), "secs must be an integer!"
    second = secs % 60
    secs //= 60
    minute = secs % 60
    secs //= 60
    hour = secs % 24
    secs //= 60
    day = secs

    return f"{day}days {hour:02}:{minute:02}:{second:02}"
