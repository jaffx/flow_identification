import os


def InitPath(path):
    if not os.path.isdir(path):
        print(f"创建文件夹{path}")
        os.makedirs(path)


SourcePath = "/Users/lyn/codes/python/Flow_Identification/Dataset/v1/WMS_Source_B"
SimplePath = "/Users/lyn/codes/python/Flow_Identification/Dataset/v1/WMS_Simple_B"

InitPath(SimplePath)

SrcTrain, SrcVal = os.path.join(SourcePath, "train"), os.path.join(SourcePath, "val")
SimTrain, SimVal = os.path.join(SimplePath, "train"), os.path.join(SimplePath, "val")

InitPath(SimTrain)
InitPath(SimVal)

for cls in os.listdir(SrcTrain):
    if cls.startswith('.'):
        continue
    src_cls = os.path.join(SrcTrain, cls)
    sim_cls = os.path.join(SimTrain, cls)
    InitPath(sim_cls)
    for file in os.listdir(src_cls):
        if file.startswith('.'):
            continue
        print(f"处理文件{file}")
        src_file = os.path.join(src_cls, file)
        sim_file = os.path.join(sim_cls, file)
        with open(src_file) as rfp, open(sim_file, 'w+') as wfp:
            count = 0
            lines = rfp.readlines()[2:]
            for line in lines:
                line = line.strip()
                items = line.split(" ")
                value = float(items[-1])
                wfp.write(f"{value:.4f}\n")
                count += 1
            rfp.close()
            wfp.close()
            print(f"处理行数:{count}")

for cls in os.listdir(SrcVal):
    if cls.startswith('.'):
        continue
    src_cls = os.path.join(SrcVal, cls)
    sim_cls = os.path.join(SimVal, cls)
    InitPath(sim_cls)
    for file in os.listdir(src_cls):
        if file.startswith('.'):
            continue
        print(f"处理文件{file}")
        src_file = os.path.join(src_cls, file)
        sim_file = os.path.join(sim_cls, file)
        with open(src_file) as rfp, open(sim_file, 'w+') as wfp:
            count = 0
            lines = rfp.readlines()
            for line in lines:
                line = line.strip()
                items = line.split(" ")
                value = float(items[-1])
                wfp.write(f"{value:.4f}\n")
                count += 1
            rfp.close()
            wfp.close()
            print(f"处理行数:{count}")
