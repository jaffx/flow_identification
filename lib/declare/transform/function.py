"""
操作全局transform的方法接口
"""
__declare_transforms = {

}


def addTransform(name, creator, desc):
    global __declare_transforms

    assert name not in __declare_transforms, f"{name}-transform already exists!"
    assert creator is not None, "transform can't be empty!"
    assert desc, "desc can't be empty!"
    __declare_transforms[name] = {
        "creator": creator,
        "desc": desc
    }


def getTransform(name):
    global __declare_transforms

    assert name in __declare_transforms, f"{name}-transform not exist"
    return __declare_transforms[name]["creator"]()


def getTransformDesc(name):
    global __declare_transforms

    assert name in __declare_transforms, f"{name}-transform not exist"
    return __declare_transforms[name]["desc"]


def getAllTransformInfos():
    global __declare_transforms

    return [{"name": key, "desc": getTransformDesc(key)} for key in __declare_transforms]
