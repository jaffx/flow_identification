import os


class Analyzer():
    def __init__(self, path):
        self.path = path

    def Analyse_train_iter(self, attrs=[]):
        """
        file_format:
        ['Method', ' epoch', ' batch', 'NSample', 'AccNum', '   ACC', '  Loss', 'AVGLoss\n']
        ['     T', '     0', '     1', '    32', '    10', '0.3125', '  1.39', '0.0433\n']
        ['     T', '     0', '     2', '    32', '     9', '0.2812', '  1.37', '0.0428\n']
        """
        file_path = os.path.join(self.path, "train_iter")
        with open(file_path) as fp:
            content = fp.readlines()
            results = {
                "method": [],
                "epoch": [],
                "batch": [],
                "sample_num": [],
                "acc_num": [],
                "acc": [],
                "loss": [],
                "avg_loss": [],
            }
            for line in content[1:]:
                line = line[:-1]
                datas = line.split('\t')
                method = datas[0].strip()
                epoch = int(datas[1])
                batch = int(datas[2])
                sample_num = int(datas[3])
                acc_num = int(datas[4])
                acc = float(datas[5])
                loss = float(datas[6])
                avg_loss = float(datas[7])
                results["method"].append(method)
                results["epoch"].append(epoch)
                results["batch"].append(batch)
                results["sample_num"].append(sample_num)
                results["acc_num"].append(acc_num)
                results["acc"].append(acc)
                results["loss"].append(loss)
                results["avg_loss"].append(avg_loss)
            ret = {}
            for attr in attrs:
                ret[attr] = results[attr]
            return ret
