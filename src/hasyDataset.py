# CSCI 5525 | Group 11
# Written by Alex
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

import data_utils

class HasyDataset(Dataset):
    def __init__(self, images=None, labels=None, path=None):
        if images is None and labels is None:
            images, labels, symbols = data_utils.load_hasy(csv_filepath=path)
            self.images = images
            self.labels = [data_utils.get_hasy_label(labels,i,symbols) for i in range(len(images))]
        else:
            self.images = images
            self.labels = labels
        self.transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.ToTensor(), 
                transforms.Normalize((0.1307,), (0.3081,))
            ])
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        return image, label

if __name__ == "__main__":
    path = "C:\\Users\\pivin\\OneDrive\\Documents\\repos\\CSCI-5525-Project\\src\\data\\hasy\\classification-task\\fold-1\\test.csv"
    data = data_utils.get_hasy_loaders(filepath=path)

    print(data)