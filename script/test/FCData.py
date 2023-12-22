from model.Dataset.FCDataset import FCDataset
import json
dataset = FCDataset(path="/Users/lyn/codes/python/Flow_Identification/Dataset/fc/val", length=10, step=5)

print(json.dumps(dataset.getData(2),indent=4))