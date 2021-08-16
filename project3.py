import torch
import torchvision.models as models
from  torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#METHODS-----------------------------------------------
def train_model(image_datasets, model, criterion, optimizer, scheduler, num_epochs):
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
    batch_size = 1, shuffle = True, num_workers = 4) for x in ['train', 'valid']}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_no_corrects = 0
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in dataloaders['train']: # iterate over data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        model.eval()
        no_corrects = 0
        for inputs, labels in dataloaders['valid']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                no_corrects += torch.sum(preds == labels.data)
        if no_corrects > best_no_corrects:
            best_no_corrects = no_corrects
            best_model_wts = copy.deepcopy(model.state_dict())
        scheduler.step()
    model.load_state_dict(best_model_wts)
    return model

def acc(c1,c2,c3,f1,f2,f3):
    print("class1 acc: " , (c1/(c1+f1)))
    print("class2 acc: " , (c2/(c2+f2)))
    print("class3 acc: " , (c3/(c3+f3)))
    print("overall:", (c1+c2+c3)/(c1+c2+c3+f1+f2+f3))
#----------------------------------------------------------------------

#Construction of model and its parameters------------------------------
model_conv = models.alexnet(pretrained = True)
for param in model_conv.parameters():
  param.requires_grad = False

model_conv.classifier[ 6 ] = nn.Linear(4096,3)
model_conv = model_conv.to(device) # use the GPU
criterion = nn.CrossEntropyLoss()
optimizer_conv = optim.SGD(model_conv.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR( optimizer_conv, step_size = 1, gamma =  0.1)

data_dir = "drive/MyDrive/data/"
data_transforms = {
 'train': transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
 ]),
 'valid':  transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
 ]),
 'test': transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
 ]),
}
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
 data_transforms[x]) for x in ['train', 'valid', 'test']}
#--------------------------------------------------------------------------

#Training------------------------------------------------------------------
model = train_model(image_datasets, model_conv, criterion,optimizer_conv,exp_lr_scheduler,20)

#--------------------------------------------------------------------------

## Show test performance---------------------------------------------------
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size = 1, shuffle = False, num_workers = 3)for x in ['test']}
correct1, correct2, correct3, false1, false2, false3, sample, no_corrects = 0,0,0,0,0,0,0,0
for inputs, labels in dataloaders['test']:
    inputs = inputs.to(device)
    sample += 1
    labels = labels.to(device)
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    if int(labels.data.item()) ==0:
        if int(preds.item())==0:
            correct1 +=1
        else:
            false1+=1
    if int(labels.data.item()) ==1:
        if int(preds.item())==1:
            correct2 +=1
        else:
            false2+=1
    if int(labels.data.item()) ==2:
        if int(preds.item())==2:
            correct3+=1
        else:
            false3+=1
    no_corrects += torch.sum(preds == labels.data)
    print(acc(correct1, correct2, correct3, false1, false2, false3))