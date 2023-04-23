# CSCI 5525 | Group 11
# Written by Alex

import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class crohmeDataset(Dataset):
    def __init__(self, image_folder=os.getcwd()+"\\src\\data\\crohme\\images", label_file=os.getcwd()+"\\src\\data\\crohme\\CROHME_math.txt", transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])):
        """
        Make sure you unzip the files first.

        Args:
            - image_folder (str): path to folder of crohme images
            - label_file (str): path to labels .txt
            - transform: transform list of normalizing values for images
        """
        self.image_folder = image_folder
        self.label_file = label_file
        self.transform = transform
        with open(label_file, 'r') as f:
            self.labels = [line.strip() for line in f.readlines() if line != ""]

        self.image_files = sorted(os.listdir(image_folder))

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        filename = self.image_files[idx]
        img_path = os.path.join(self.image_folder, filename)
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        label = self.labels[idx]
        return img, label, filename