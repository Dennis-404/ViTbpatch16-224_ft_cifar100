from logging import debug
import math
from safetensors.torch import load_file
from utils.utils import get_logger
from torch.utils.data import DataLoader
import torch    
import torch.nn.functional as F
import timm
import os
import copy
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset
import torchvision.models as models
import torch.utils.data
import torch.optim as optim  
from torch.utils.data import DataLoader  
from torchvision import datasets, transforms, models  
from torch.optim.lr_scheduler import StepLR  

class CustomCIFAR100(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label



def train_model_oncifar100(model):  
    # hyper  
    batch_size = 128  
    num_epochs = 60
    modelarch='ViTb16cifar100'

    transform = transforms.Compose(
        [transforms.Resize(256),           #transforms.Scale(256)
        transforms.CenterCrop(224), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    
    train_dataset = datasets.CIFAR100(root='dataset', train=True, download=False, transform=transform)  
    test_dataset = datasets.CIFAR100(root='dataset', train=False, download=False, transform=transform)  
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=4)  
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,num_workers=4)  

    device = torch.device("cuda:5")  
    model = model.to(device)   

    ls=0.1
    criterion = nn.CrossEntropyLoss(label_smoothing=ls)
    lr=0.01
    optimizer = optim.AdamW(model.parameters(), lr=lr)  

    accs=[]
    accs.append(0)

    # train  
    for epoch in range(num_epochs):  
        model.train()  
        running_loss = 0.0  
        for i, data in enumerate(train_loader, 0):  
            inputs, labels = data[0].to(device), data[1].to(device)  

            optimizer.zero_grad()  
 
            outputs= model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()  
            optimizer.step()
  
            running_loss += loss.item()  
            if i % 100 == 0:  
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))  
                running_loss = 0.0  
        # eval  
        model.eval()  
        val_loss = 0.0  
        correct = 0
        correctbranch=0  
        total1=0
        with torch.no_grad(): 
            for data in test_loader:  
                images, labels = data[0].to(device), data[1].to(device)  
                outputs = model(images)
 
                probabilities = F.softmax(outputs, dim=1)  
                _, predicted = torch.max(probabilities, 1)  

                total = labels.size(0)
                total1+=total
                correct +=(predicted == labels).sum().item()
                #correctbranch +=(predictedbranch == labels).sum().item()  
            print('Accuracy of the network on val: %d %%' % (100 * correct / total1))
            acc=100 * (correct / total1)
        if acc>max(accs):    
            accs.append(acc)
            model.eval()
            print('saving checkpoint')
            filename = f"models/best_{modelarch}_{acc}_lr{lr}_ls{ls}.pth"
            torch.save(model.state_dict(),filename)
        else:
            accs.append(acc)
    
    return


if __name__ == '__main__':

    ckpt = "model.safetensors"
    model = timm.create_model('vit_base_patch16_224', pretrained=False)
    model.load_state_dict(load_file(ckpt))
    model.head=nn.Linear(model.embed_dim,100)

    train_model_oncifar100(model)
