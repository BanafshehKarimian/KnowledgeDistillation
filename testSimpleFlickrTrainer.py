from torch.utils.data import DataLoader, Dataset
from Data.FlickrLoader import Data
import torch
from Models.ResNet50Model import ResNet50Class
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
from Trainers.simple import SimpleTrainer

full_dataset = Data("Data/flickr8k/")
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=8)
test_dataloader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=8)
device = "cuda" if torch.cuda.is_available() else "cpu"
learner_model = ResNet50Class()
learner = SimpleTrainer(learner_model, device)
num_epochs = 3
print("acc before training:")
print(learner.eval(test_dataloader))
learner.train(num_epochs, train_dataloader)
print("acc after training:")
print(learner.eval(test_dataloader))