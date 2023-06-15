import os


def InitPath(path):
    if not os.path.isdir(path):
        print(f"创建文件夹{path}")
        os.makedirs(path)


SourcePath = "/Users/lyn/codes/python/Flow_Identification/Dataset/v2/WMS/v2_WMS_Label_Source_A"
SimplePath = "/Users/lyn/codes/python/Flow_Identification/Dataset/v2/WMS/v2_WMS_Label_Simple_A"

InitPath(SimplePath)

for cls in os.listdir(SourcePath):
    if cls.startswith('.'):
        continue
    src_cls = os.path.join(SourcePath, cls)
    sim_cls = os.path.join(SimplePath, cls)
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

