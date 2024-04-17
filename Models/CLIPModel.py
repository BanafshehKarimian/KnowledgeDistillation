from torchvision import models
import torch
import torch.optim as optim
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel, AdamW, get_scheduler, CLIPConfig
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from datasets import load_metric

class CLIPClass():
    def __init__(self):
        config = CLIPConfig.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel(config)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def forward(self, batch, return_loss = True):
        return self.model(**batch, return_loss = return_loss)