# CSCI 5525 - Group 11

################################
#           Modules            #
################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_utils import *
import os
import CROHMEDataset

################################
#             Main             #
################################

if __name__ == "__main__":
    train_loader, test_loader, val_loader = CROHMEDataset.get_CROHME_loaders()

    # Get a sample from the train_loader
    data_iter = iter(train_loader)
    sample_images, sample_labels = next(data_iter)

    # Display the first image and its label
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    CROHMEDataset.print_sample(sample_images[0], mean, std)
    print("Label:", sample_labels[0])