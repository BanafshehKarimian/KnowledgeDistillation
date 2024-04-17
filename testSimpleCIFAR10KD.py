from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import torch
from Models.ResNet50Model import ResNet50Class
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
from Trainers.simple import SimpleTrainer, SimpleKDTrainer
from torchvision.transforms import ToTensor

training_data = datasets.CIFAR10(
    root="Data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.CIFAR10(
    root="Data",
    train=False,
    download=True,
    transform=ToTensor()
)
train_dataloader = DataLoader(dataset=training_data, shuffle=True, batch_size=8)
test_dataloader = DataLoader(dataset=test_data, shuffle=True, batch_size=8)
device = "cuda" if torch.cuda.is_available() else "cpu"
teacher_model = ResNet50Class(class_num = 10)
student_model = ResNet50Class(class_num = 10)
lambda_param = 0.5
learner = SimpleTrainer(teacher_model, device)
num_epochs = 3
'''Teacher pre-training'''
print("teacher acc before training:")
print(learner.eval(test_dataloader, dtype = "list", x_label = 0, y_label = 1))
learner.train(num_epochs, train_dataloader, dtype = "list", x_label = 0, y_label = 1)
print("teacher acc after training:")
print(learner.eval(test_dataloader, dtype = "list", x_label = 0, y_label = 1))
'''Training the student'''
KDlearner = SimpleKDTrainer(teacher_model, student_model, lambda_param, device)
print("student acc before training:")
print(KDlearner.eval(test_dataloader, dtype = "list", x_label = 0, y_label = 1))
KDlearner.offline_train(num_epochs, train_dataloader, dtype = "list", x_label = 0, y_label = 1)
print("student acc after training:")
print(KDlearner.eval(test_dataloader, dtype = "list", x_label = 0, y_label = 1))