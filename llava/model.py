from transformers import LlavaForConditionalGeneration, LlavaProcessor
from torchvision.datasets import Caltech101
from torch import nn
import torch
import pickle
import json
import random

class LlavaClassifier():
    def __init__(self, categories):
        self.model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", device_map="cuda", torch_dtype=torch.bfloat16)
        self.processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", use_fast=False)
        
        self.categories = categories
        self.categories_map = {category:i for i, category in enumerate(categories)}
        
        self.prompt = """You are an expert in image classification. First, pay attention to details describe the image thoroughly. Then, based on your description and the image, classify the main object in this image by choosing from a following list of categories:

[CATEGORIES]

Lastly, output a JSON object in one line as the final line of your output. Your category absolutely MUST be in the list and your output MUST also be a JSON object and adhere to the following syntax:

{ "category": "[YOUR SELECTION]" }"""
    
    def generate_prompt(self):
        random.shuffle(self.categories)
        prompt = self.prompt.replace("[CATEGORIES]", "\n".join(self.categories))
        return [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }]
    
    def __call__(self, x):
        prompts = [self.processor.apply_chat_template(self.generate_prompt(), add_generation_prompts=True) for _ in range(len(x))] 
        inputs = self.processor(images=x, text=prompts, padding=True, return_tensors="pt").to("cuda")
        generate_ids = self.model.generate(**inputs, max_new_tokens=30)
        outputs = self.processor.batch_decode(generate_ids, skip_special_tokens=True)
        outputs = [output.split("\n")[-1].strip() for output in outputs]
        
        onehot_outputs = []
        for output in outputs:
            onehot = torch.zeros((len(self.categories)))
            try:
                category = json.loads(output)["category"]
                onehot[self.categories_map[category]] = 1
            except:
                pass
            onehot_outputs.append(onehot)
        
        return torch.vstack(onehot_outputs)
