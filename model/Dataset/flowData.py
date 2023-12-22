from lib.xyq import printer


class flowData:
    """
    保存一条流型数据
    """

    def __init__(self, data=None, label=None, file_path=None):
        self.r_ptr = 0

        self.data = data
        self.label = label  # label类型必须是int类型
        self.file_path = file_path

    def __len__(self):
        return len(self.data)

    def __call__(self, step):
        return self.getSample(step)

    def getSample(self, length, step):
        assert length <= len(self.data), f"The length of sample {length} must less than flow data {len(self.data)}"
        assert step > 0, f"The step {step} is invalid which must greater than 0"
        end = self.r_ptr + length
        if end >= len(self.data):
            self.r_ptr = 0
        # data = copy.deepcopy(self.data[self.r_ptr:end])
        data = self.data[self.r_ptr:end]
        self.r_ptr += step
        return data, self.label, self.file_path

    def loadData(self, label=-1):
        try:
            with open(self.file_path) as fp:
                content = fp.readlines()
                content = content[2:]
                data = []
                for line in content:
                    data.append(float(line))
                self.data = data
                self.label = label
        except Exception as e:
            printer.xprint_red(f"{self.file_path} 加载错误，原因 {e}", end="\n\n")
            exit(1)
            return None

    def Init(self):
        self.r_ptr = 0

    def isReadableForLength(self, length):
        return self.r_ptr + length < len(self.data)


def readWMSFile(data_path, cls_name):
    """
    从文件中加载WMS数据，适用于.epst文件
    @param dataset_path: 数据集地址
    @param cls_name: 类名
    @param filename: 文件名
    @return
    """
    file_path = data_path
    try:

        with open(file_path) as fp:
            content = fp.readlines()
            content = content[2:]
            data = []
            for line in content:
                line = line.strip()
                items = line.split(" ")
                data.append(float(items[-1]))
        return flowData(data, cls_name, file_path)
    except Exception as e:
        printer.xprint_red(f"{file_path} 加载错误，原因 {e}")
        return None


def readSimpleData(data_path, label):
    file_path = data_path
    try:
        with open(file_path) as fp:
            content = fp.readlines()
            content = content[2:]
            data = []
            for line in content:
                data.append(float(line))
        return flowData(data, label, file_path)
    except Exception as e:
        printer.xprint_red(f"{file_path} 加载错误，原因 {e}", end="\n\n")
        exit(1)
        return None
