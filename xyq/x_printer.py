import os.path
import datetime
import logging

TARGET_FILES = 'console_log.txt'


def write_message(func):
    def write_inner(*args, **kwargs):
        func(*args, **kwargs)

        with open(TARGET_FILES, 'a+') as fp:
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
def xprint_yellow(*args, **kwargs):
    print("\033[33m", *args, "\033[0m", **kwargs)


@write_message
def xprint_blue(*args, **kwargs):
    print("\033[34m", *args, "\033[0m", **kwargs)


@write_message
def xprint_purple(*args, **kwargs):
    print("\033[35m", *args, "\033[0m", **kwargs)


@write_message
def xprint_cyan(*args, **kwargs):
    print("\033[35m", *args, "\033[0m", **kwargs)


@write_message
def xprint(*args, **kwargs):
    print(*args, **kwargs)
