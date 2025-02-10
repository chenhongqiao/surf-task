from torchvision.datasets import Caltech101
import random
import pickle
from os import path
import os

dataset = list(Caltech101("./data", download=True))

random.seed(123)
random.shuffle(dataset)

test_set = dataset[:1000]
validation_set = dataset[1000:2000]
train_set = dataset[2000:]

BASE_PATH = "./data/caltech101_splits"
if not path.exists("./data/caltech101_splits"):
    os.makedirs("./data/caltech101_splits")

with open(path.join(BASE_PATH, "test.pkl"), "wb") as f:
    pickle.dump(test_set, f)

with open(path.join(BASE_PATH, "validation.pkl"), "wb") as f:
    pickle.dump(validation_set, f)

with open(path.join(BASE_PATH, "train.pkl"), "wb") as f:
    pickle.dump(train_set, f)