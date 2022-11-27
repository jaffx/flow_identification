from DataLoader.flowDataset import flowDataset


class flowDataLoader():
    dataset = None
    dprate = 0
    batch_size = 0

    def __init__(self, dataset: flowDataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size

    def getData(self):
        data = self.dataset.getData(self.batch_size)
