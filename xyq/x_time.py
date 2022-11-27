import time
import xyq.x_printer as printer


def showTimeMs(func):
    def getTimeMsInner(*args, **kwargs):
        time0 = time.time()
        func(*args, **kwargs)
        printer.xprint(f"Function<{func.__name__}{args}{kwargs}>running time: {int((time.time() - time0) * 1000)}ms")

    return getTimeMsInner


@showTimeMs
def test(x,y,z):
    time.sleep(2)

test(2,2,z = 2)