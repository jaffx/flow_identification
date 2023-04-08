import os


def is_train(path):
    files = os.listdir(path)
    return "epoch" in files


def is_yaml(path):
    files = os.listdir(path)
    if "info.yaml" in files:
        return True
    elif "info" in files:
        return False



class analyzer:
    def __init__(self):
        pass
