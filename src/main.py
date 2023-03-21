# CSCI 5525 - Group 11

################################
#           Modules            #
################################
import numpy as np
import pandas as pd

import sys
import logging # for console output logging

import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader

# custom classes
from MLP import MLP
from HAZYDataset import HAZYDataset

################################
#        Load HASYv2(s)        #
################################

# Be sure to unzip the .tar before running

# Load training data
## Note for the file path there are 10 folders (fold-1 through fold-10)
## each which have train.csv and test.csv
csv_filepath = "src\\data\\classification-task\\fold-1\\test.csv" # may need to be modified depending on your environment

# load data using HAZYDataset.py (ask Alex)
test_data = HAZYDataset(csv_filepath)
test_loader = DataLoader(test_data, batch_size=32)

#TODO: train_loader, test_loader

################################
#        Train models          #
################################

# Hyperparameters - CONSTANT
input_size = 32 * 32  # images are 32x32 pixels
hidden_size = 128     # hidden layer nodes
output_size = 369     # 369 possible labels

# Hyperparameters - MODIFY THESE
num_epochs = 10             
learning_rate = [1e-2]       
descent_optimizers = [torch.optim.SGD] # SGD, Adagrad, RMSprop, Adam 

try:
    sys.stdout = open("output.txt","w")
    for optim in descent_optimizers:
        for eta in learning_rate:
            print("Currently running " + optim.__name__ + " at eta = %.1E" % (eta)) 

            # Instantiate the MLP model
            model = MLP(input_size, hidden_size, output_size, eta, num_epochs)

            # Define the loss function and the optimizer
            criterion = nn.CrossEntropyLoss()

            # As given above
            optimizer = optim(model.parameters(), lr=eta)

            # Train the model
            loss_history, error_rate_history = model.fit(train_loader, criterion, optimizer)

            # Evaluate the model on the test set
            test_loss, test_error_rate = model.predict(test_loader, criterion)
            print("\n\n")
finally:
    print("DONE")
    sys.stdout.close()
    sys.stdout = sys.__stdout__