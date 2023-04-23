# CSCI 5525 | Group 11
# Written by Alex

# An initial Convolutional Neural Network for simple image classification
# using the HASYv2 dataset
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
# Using pytorch?

class CNN(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride,
                 max_pool_size, learning_rate, max_epochs):
        """Constructor, initialize class parameters and layers."""
        super().__init__()

        # Hyperparameters
        self.lr = learning_rate
        self.max_epochs = max_epochs

        # CNN architecture
        self.conv_layer = nn.Conv2d(input_size, output_size, kernel_size)
        self.pool = nn.MaxPool2d(max_pool_size,stride)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(3380, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, X):
        """Propagate forward"""
        X = self.conv_layer(X)    # convolutional layer
        X = nn.ReLU()(X)          # ReLU activation
        X = self.pool(X)          # max-pooling layer  
        X = self.dropout(X)       # dropout layer
        X = X.view(X.size(0), -1) # flatten input
        X = self.fc1(X)           # first f-c layer
        X = nn.ReLU()(X)          # ReLU activation
        X = self.fc2(X)           # second f-c layer
        X = nn.Softmax(dim=1)(X)  # apply soft-max activation 
        
        return X

    def fit(self, train_loader, criterion, optimizer):
        """Train and fit"""
        self.train()

        # lists for saving loss/error history
        losses = []
        error_rates = []
        
        # for convergence check
        prev_epoch_loss = None

        # Epoch loop
        for i in range(self.max_epochs):
            # cumulative variables for loss/error
            loss_sum = 0.0
            correct = 0
            total = 0

            # Mini batch loop
            for j,(images,labels) in enumerate(train_loader):
                images = images.float()

                # Forward pass (consider the recommmended functions in homework writeup)
                outputs = self.forward(images)
            
                # Backward pass and optimize (consider the recommmended functions in homework writeup)
                # Make sure to zero out the gradients using optimizer.zero_grad() in each loop
                optimizer.zero_grad()
                labels = torch.tensor(labels)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Track the loss and error rate
                loss_sum += loss.item()
                total += labels.size(0)

                _, predicted = torch.max(outputs.data, 1) # get index of predicted images
                # count correct labels
                for k in range(len(predicted)):
                    if predicted[k] == labels[k]:
                        correct += 1

                
            # Print/return training loss and error rate in each epoch
            epoch_loss = loss_sum / len(train_loader)
            epoch_error_rate = 1 - correct / total

            # add to history lists
            losses.append(epoch_loss)
            error_rates.append(epoch_error_rate)

            # print the loss and error rate for the current epoch
            print("Epoch [%d/%d]: Loss: %.4f | Error: %.4f" \
                  % ((i+1), self.max_epochs, epoch_loss, epoch_error_rate))
            
            # check for convergence, arbitarily chose 1e-4
            if prev_epoch_loss and abs(prev_epoch_loss - epoch_loss) < 1e-4:
                break
            prev_epoch_loss = epoch_loss

        return losses, error_rates

    def predict(self, test_loader, criterion):
        self.eval()
        """Predict data"""
        # initialize counters
        loss = 0.0
        correct = 0
        total = 0

        # track incorrect predictions
        incorrect = []
        
        with torch.no_grad(): # no backprop step so turn off gradients
            for j,(images,labels) in enumerate(test_loader):
                # Compute prediction output and loss
                outputs = self.forward(images)

                # Track the loss and error rate
                labels = torch.tensor(labels)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1) # get index of predicted images
                total += labels.size(0)
                
                # count correct labels
                for k in range(len(predicted)):
                    if predicted[k] == labels[k]:
                        correct += 1
                    else:
                        incorrect.append((images[k],predicted[k],labels[k])) # incorrect item (image, predicted, correct)

        # Print/return test loss and error rate
        loss = loss / len(test_loader)
        test_error_rate = 1 - correct / total
        print("Test Loss: %.4f | Test Error: %.4f" % (loss, test_error_rate))

        return loss, test_error_rate, incorrect

if __name__ == "__main__":
    import sys
    import data_utils
    path = "C:\\Users\\pivin\\OneDrive\\Documents\\repos\\CSCI-5525-Project\\src\\data\\hasy\\classification-task"
    fold = "\\fold-1"
    filename = "\\test.csv"

    # images, labels, symbol_index = data_utils.load_hasy(csv_filepath=f"{path}{fold}{filename}")
    # data_utils.plot_hasy_img(images[0])
    # id = data_utils.get_hasy_label(labels, 0, symbol_index)
    # print(id)
    
    # train_loader = data_utils.get_hasy_loaders(filepath=f"{path}{fold}\\train.csv")
    # test_loader = data_utils.get_hasy_loaders(filepath=f"{path}{fold}\\test.csv")
    train_loader = data_utils.get_hasy_loaders(filepath=f"{path}{fold}\\test.csv")
    test_loader = train_loader

    # images, labels = next(iter(test_loader))
    # # # print(images[0])
    # for img,lbl in zip(images, labels):
    #     data_utils.plot_hasy_img(img)
    #     print(lbl) 

    # Hyperparameters - CONSTANT
    input_size = 1  
    output_size = 20 
    kernel_size = 3
    stride_size = None # dunno what to do with this
    max_pool_size = 2

    # Hyperparameters - MODIFY THESE
    max_epochs = 10              
    learning_rate = 0.1
    
    try:
        sys.stdout = open("hasyCNN.txt","w")
        # Instantiate the MyCNN class
        model = CNN(input_size, output_size, kernel_size, stride_size, \
                    max_pool_size, learning_rate, max_epochs)

        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        # Train the model
        loss_history, error_rate_history = model.fit(train_loader, criterion, optimizer)

        # Test the model
        test_loss, test_error_rate, incorrect = model.predict(test_loader, criterion)

        print("\n")

    finally:
        print("DONE")
        sys.stdout.close()
        sys.stdout = sys.__stdout__
