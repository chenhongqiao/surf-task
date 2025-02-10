from torch import nn
from transformers import CLIPVisionModel, CLIPImageProcessor
from pathos.multiprocessing import Pool

class CLIPClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        self.clf_head = nn.Linear(1024, 101)
        
        for p in self.encoder.parameters():
            p.requires_grad = False
        
    def forward(self, x):
        embedding = self.encoder(x).pooler_output
        return self.clf_head(embedding)

def clip_preprocess(dataset):
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    def collate_fn(x):
        image = processor(images=x[0], return_tensors="pt")['pixel_values'].squeeze()
        label = x[1]
        return (image, label)
    
    with Pool(processes=32) as pool:
        dataset = pool.map(collate_fn, dataset)
    
    return dataset
