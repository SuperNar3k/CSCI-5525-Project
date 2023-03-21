# CSCI 5525 - Group 11

# Be sure to unzip the .tar before running
# help(hasy) # prints some helpful tools, somewhat deprecated?
import numpy as np
import matplotlib.pyplot as plt
import torch

from data import hasy_tools as hasy

class HAZYDataset:
    def __init__(self, csv_filepath) -> None:
        """
        Initializes the HAZYDataset
        
        Args:
            csv_filepath (str): filepath to the appropriate csv file to be loaded
        """
        print(f"Loading data from: '{csv_filepath}'")
        self.symbol_index = hasy.generate_index(csv_filepath)                       # returns dict of {label : index} pairs
        self.images, self.labels = hasy.load_images(csv_filepath,self.symbol_index) # images is ndarray of size (index,y,x,depth)
        print(f"Loaded {len(self)} images...")

    def __len__(self):
        """Returns the amount of images"""
        return len(self.images)

    def __getitem__(self, idx):
        """
        Get an image and label from given index or indices
        
        Args:
            idx (int or torch.tensor): index of data to be returned

        Returns:
            (torch.Tensor, torch.Tensor): tensor representations of the given data
        """
        return torch.Tensor(self.images[idx]), torch.Tensor(self.labels[idx])
    
    def load_data(self):
        """
        Turn the loaded data into a Pytorch data loader
        
        Returns:

        """
    def print_image(self, idx):
        """
        Plots a 32x32 black & white image from given index
        
        Args:
            idx: index of image to be printed
        """
        sample = hasy.thresholdize(self.images[idx]) 
        plt.imshow(sample, cmap='gray')                
        plt.show()