import os.path
import datetime
import logging

TARGET_FILES = ['console_log.txt']


def write_message(func):
    def write_inner(*args, **kwargs):
        func(*args, **kwargs)
        for file in TARGET_FILES:
            with open(file, 'a+') as fp:
                for msg in args:
                    fp.write(msg)
                fp.write('\n')
                fp.close()

    return write_inner


@write_message
def xprint_green(*args, **kwargs):
    print("\033[32m", *args, "\033[0m", **kwargs)


@write_message
def xprint_red(*args, **kwargs):
    print("\033[31m", *args, "\033[0m", **kwargs)


@write_message
def xprint(*args, **kwargs):
    print(*args, **kwargs)
