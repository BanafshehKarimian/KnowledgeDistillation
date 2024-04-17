import json
from PIL import Image as Img

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import clip
from transformers import CLIPProcessor, CLIPModel, get_scheduler, CLIPConfig
import numpy as np
import string 
import matplotlib.image as mpimg
import pandas as pd
from torch.utils.data import Dataset
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
from datasets import load_metric
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from torchvision.transforms import ToTensor
import torchvision.transforms as T
import torch.optim as optim
from sklearn.metrics import confusion_matrix

class SimpleTrainer():
    def __init__(self, model, device, optim_lr = 5e-5):
        self.model = model
        self.device = device
        self.model.to(self.device)        
        self.optimizer = AdamW(self.model.parameters(), lr = optim_lr)
        

    def compute_loss(self, y, yp):
        return F.cross_entropy(y, yp)

    def train(self, num_epochs, train_dataloader, dtype = "map", x_label = "pixel_values", y_label = "input_ids"):
        num_training_steps = num_epochs * len(train_dataloader)
        progress_bar = tqdm(range(num_training_steps))
        lr_scheduler = get_scheduler(
                    "linear",
                    optimizer=self.optimizer,
                    num_warmup_steps=0,
                    num_training_steps=num_training_steps
                )
        self.model.train()
        for epoch in range(num_epochs):
            for batch in train_dataloader:
                if dtype == "map":
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                x = batch[x_label].to(self.device)
                y = batch[y_label].to(self.device)
                outputs = self.model(x)
                loss = self.compute_loss(outputs, y)
                loss.backward()
                self.optimizer.step()
                lr_scheduler.step()
                self.optimizer.zero_grad()
                progress_bar.update(1)
    

    def eval(self, eval_dataloader, dtype = "map", x_label = "pixel_values", y_label = "input_ids", metric = "accuracy"):
        metric= load_metric(metric)
        self.model.eval()
        for batch in eval_dataloader:
            if dtype == "map":
                batch = {k: v.to(self.device) for k, v in batch.items()}
            x = batch[x_label].to(self.device)
            y = batch[y_label].to(self.device)
            with torch.no_grad():
                outputs = self.model(x)
            metric.add_batch(predictions=outputs.argmax(1), references=y)
        return metric.compute()

class SimpleKDTrainer():
    def __init__(self, teacher_model, student_model, lambda_param, device):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.device = device
        self.lambda_param = lambda_param
        self.teacher_model.to(self.device)
        self.student_model.to(self.device)
        self.optimizer = AdamW(self.student_model.parameters(), lr=5e-5)
        self.loss_function = nn.KLDivLoss(reduction="batchmean")

    def compute_student_target_loss(self, y, yp):
        return F.cross_entropy(y, yp)
    
    def compute_distillation_loss(self, student_logits, teacher_logits):
        return self.loss_function(student_logits, teacher_logits.log()) * 25
        
    def offline_train(self, num_epochs, train_dataloader, dtype = "map", x_label = "pixel_values", y_label = "input_ids"):
        num_training_steps = num_epochs * len(train_dataloader)
        progress_bar = tqdm(range(num_training_steps))
        lr_scheduler = get_scheduler(
                    "linear",
                    optimizer=self.optimizer,
                    num_warmup_steps=0,
                    num_training_steps=num_training_steps
                )
        self.teacher_model.eval()
        self.student_model.train()
        for epoch in range(num_epochs):
            for batch in train_dataloader:
                if dtype == "map":
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                x = batch[x_label].to(self.device)
                y = batch[y_label].to(self.device)
                student_output = self.student_model(x)
                with torch.no_grad():
                    teacher_output = self.teacher_model(x)
                student_target_loss = self.compute_student_target_loss(student_output, y)
                distillation_loss = self.compute_distillation_loss(student_output, teacher_output)
                loss = (1. - self.lambda_param) * student_target_loss + self.lambda_param * distillation_loss
                loss.backward()
                self.optimizer.step()
                lr_scheduler.step()
                self.optimizer.zero_grad()
                progress_bar.update(1)

    def eval(self, eval_dataloader, dtype = "map", x_label = "pixel_values", y_label = "input_ids", metric = "accuracy"):
        metric= load_metric(metric)
        self.student_model.eval()
        for batch in eval_dataloader:
            if dtype == "map":
                batch = {k: v.to(self.device) for k, v in batch.items()}
            x = batch[x_label].to(self.device)
            y = batch[y_label].to(self.device)
            with torch.no_grad():
                outputs = self.student_model(x)
            metric.add_batch(predictions=outputs.argmax(1), references=y)
        return metric.compute()
    
class RSNATrainer():
    def __init__(self, model, device, optim_lr = 5e-5):
        self.model = model
        self.device = device
        self.model.to(self.device)        
        self.loss_fn = nn.BCELoss()
        self.optimizer = AdamW(self.model.parameters(), lr=5e-5)
    

    def compute_loss(self, y, yp):
        return self.loss_fn(y.squeeze(), yp.float())
    
    def train(self, num_epochs, train_dataloader):
        num_training_steps = num_epochs * len(train_dataloader)
        progress_bar = tqdm(range(num_training_steps))
        lr_scheduler = get_scheduler(
                    "linear",
                    optimizer=self.optimizer,
                    num_warmup_steps=0,
                    num_training_steps=num_training_steps
                )
        self.model.train()
        loss_list = []
        for epoch in range(num_epochs):
            for batch, (img, meta_features, label) in enumerate(train_dataloader):
                img = img.to(self.device)
                meta_features = meta_features.to(self.device)
                label = label.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(img, meta_features)
                loss = self.compute_loss(outputs, label)
                loss.backward()
                loss_list.append(loss.item())
                self.optimizer.step()
                lr_scheduler.step()
                progress_bar.update(1)
        return loss_list
        
    def eval(self, eval_dataloader, metric = "accuracy"):
        #dice= load_metric("erntkn/dice_coefficient")
        self.model.eval()
        values = torch.tensor([0])
        labels = np.array([])
        predictions = np.array([])
        for batch, (img, meta_features, label) in enumerate(eval_dataloader):
            img = img.to(self.device)
            meta_features = meta_features.to(self.device)
            label = label.to(self.device)
            with torch.no_grad():
                outputs = self.model(img, meta_features)
                loss = self.compute_loss(outputs, label)
            #dice.add_batch(predictions=torch.where(outputs.squeeze(1) < 0.5, 0, 1), references=label)
            labels = np.append(labels, label.cpu().detach().numpy().ravel())
            predictions = np.append(predictions, torch.where(outputs.squeeze(1) < 0.5, 0, 1).cpu().numpy().ravel())
        labels = np.array(labels).ravel()
        predictions = np.array(predictions).ravel()
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        beta = 1
        beta_squared = beta * beta
        c_precision = tp / (tp + fp)
        c_recall = tp / (tp + fn)
        result = 0
        if c_precision > 0 and c_recall > 0:
            result = (
                (1 + beta_squared)
                * (c_precision * c_recall)
                / (beta_squared * c_precision + c_recall)
            )
        return (tp+tn)/(tp+tn+fp+fn), 2*tp/(2*tp + fp + fn), result#, dice.compute()
    
class RSNAKDTrainer():
    def __init__(self, teacher_model, student_model, lambda_param, device):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.device = device
        self.lambda_param = lambda_param
        self.teacher_model.to(self.device)
        self.student_model.to(self.device)
        self.loss_fn = nn.BCELoss()
        self.optimizer = AdamW(self.student_model.parameters(), lr=5e-5)
        self.loss_function = nn.KLDivLoss(reduction="batchmean")

    def compute_student_target_loss(self, y, yp):
        return self.loss_fn(y.squeeze(), yp.float())
    
    def compute_distillation_loss(self, student_logits, teacher_logits):
        return self.loss_function(student_logits, teacher_logits.log()) * 25
    
    def offline_train(self, num_epochs, train_dataloader):
        num_training_steps = num_epochs * len(train_dataloader)
        progress_bar = tqdm(range(num_training_steps))
        lr_scheduler = get_scheduler(
                    "linear",
                    optimizer=self.optimizer,
                    num_warmup_steps=0,
                    num_training_steps=num_training_steps
                )
        self.teacher_model.eval()
        self.student_model.train()
        for epoch in range(num_epochs):
            for batch, (img, meta_features, label) in enumerate(train_dataloader):
                img = img.to(self.device)
                meta_features = meta_features.to(self.device)
                label = label.to(self.device)
                self.optimizer.zero_grad()
                with torch.no_grad():
                    teacher_output = self.teacher_model(img, meta_features)
                student_output = self.student_model(img, meta_features)
                student_target_loss = self.compute_student_target_loss(student_output, label)
                distillation_loss = self.compute_distillation_loss(student_output, teacher_output)
                loss = (1. - self.lambda_param) * student_target_loss + self.lambda_param * distillation_loss
                loss.backward()
                self.optimizer.step()
                lr_scheduler.step()
                self.optimizer.zero_grad()
                progress_bar.update(1)

    def eval(self, eval_dataloader, metric = "accuracy"):
        #dice= load_metric("erntkn/dice_coefficient")
        self.student_model.eval()
        values = torch.tensor([0])
        labels = np.array([])
        predictions = np.array([])
        for batch, (img, meta_features, label) in enumerate(eval_dataloader):
            img = img.to(self.device)
            meta_features = meta_features.to(self.device)
            label = label.to(self.device)
            with torch.no_grad():
                outputs = self.student_model(img, meta_features)
                loss = self.compute_student_target_loss(outputs, label)
            #dice.add_batch(predictions=torch.where(outputs.squeeze(1) < 0.5, 0, 1), references=label)
            labels = np.append(labels, label.cpu().detach().numpy().ravel())
            predictions = np.append(predictions, torch.where(outputs.squeeze(1) < 0.5, 0, 1).cpu().numpy().ravel())
        labels = np.array(labels).ravel()
        predictions = np.array(predictions).ravel()
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        beta = 1
        beta_squared = beta * beta
        c_precision = tp / (tp + fp)
        c_recall = tp / (tp + fn)
        result = 0
        if c_precision > 0 and c_recall > 0:
            result = (
                (1 + beta_squared)
                * (c_precision * c_recall)
                / (beta_squared * c_precision + c_recall)
            )
        return (tp+tn)/(tp+tn+fp+fn), 2*tp/(2*tp + fp + fn), result#, dice.compute()

def pfbeta(labels, predictions, beta: float = 1):
    """
    Official implementation of the evaluation metrics, pf1 Score,
    cf. https://www.kaggle.com/competitions/rsna-breast-cancer-detection/overview/evaluation
    """
    y_true_count = 0
    ctp = 0
    cfp = 0
    for idx in range(len(labels)):

        prediction = min(max(predictions[idx], 0), 1)
        if labels[idx]:
            y_true_count += 1
            ctp += prediction
        else:
            cfp += prediction

    # Add if ever there is no true prediction to avoid divide by 0
    if y_true_count == 0:
        return 0

    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if c_precision > 0 and c_recall > 0:
        result = (
            (1 + beta_squared)
            * (c_precision * c_recall)
            / (beta_squared * c_precision + c_recall)
        )
        return result
    else:
        return 0


class CustomBCELoss(torch.nn.Module):
    def __init__(self, weight_fn=None):
        super(CustomBCELoss, self).__init__()
        self.loss_fn = torch.nn.BCELoss()
        if weight_fn is None:
            weight_fn = lambda x: 1
        self.weight_fn = weight_fn

    def forward(self, input, target):
        weight = self.weight_fn(target)
        loss = self.loss_fn(input, target)
        weighted_loss = weight * loss
        return weighted_loss.mean()