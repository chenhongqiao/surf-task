import pickle
from torchvision.datasets import Caltech101
from clip.model import CLIPClassifier, clip_preprocess
from llava.model import LlavaClassifier
from torch.utils.data import DataLoader
import torch

def evaluate_llava(data, categories):
    model = LlavaClassifier(categories)

    total_error = 0

    for i in range(0, len(data), 10):
        x, y = zip(*data[i:i+10])
        pred = torch.argmax(model(x), dim=-1)
        total_error += (pred != torch.tensor(y)).sum()

    return total_error / len(data)

def evaluate_clip(data):
    model = CLIPClassifier()
    model.clf_head = torch.load("./clip/checkpoints/10.pt", weights_only=False)
    model.to("cuda")
    
    data = clip_preprocess(data)
    x, y = zip(*data)
    x = torch.stack(x).to("cuda")
    y =  torch.tensor(y).to("cuda")
    
    pred = torch.argmax(model(x), dim=-1)
    total_error = (pred != y).sum()
    
    return total_error / len(data)

if __name__ == "__main__":
    categories = Caltech101("./data").categories

    with open("./data/caltech101_splits/test.pkl", "rb") as f:
        test_set = pickle.load(f)

    print(f"CLIP Error: {evaluate_clip(test_set):.3f}")
    print(f"LLaVA Error: {evaluate_llava(test_set, categories):.3f}")