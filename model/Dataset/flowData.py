class flowData:
    """
    保存一条流型数据
    """

    def __init__(self, data, label: int, file_path):
        self.r_ptr = 0
        self.data = data
        assert isinstance(label, int), f"label should be int instance, now the type of label is {type(label)}"
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

    def Init(self):
        self.r_ptr = 0

    def isReadableForLength(self, length):
        return self.r_ptr + length < len(self.data)
