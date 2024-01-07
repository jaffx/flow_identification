__DECLARE_FUNCS__ = {}


def addNet(name: str, creator, desc: str):
    assert name not in __DECLARE_FUNCS__, "模型重复注册"
    __DECLARE_FUNCS__[name] = (creator, desc)


def getNet(name: str):
    assert name in __DECLARE_FUNCS__, "模型不存在"
    return __DECLARE_FUNCS__[name][0]


def getAllNetInfo():
    return [{"name": name, "desc": __DECLARE_FUNCS__[name][1]} for name in __DECLARE_FUNCS__]
