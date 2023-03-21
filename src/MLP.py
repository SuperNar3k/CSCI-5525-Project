# Alex Zhu - from HW3
# CSCI 5525

import numpy as np

import torch
import torch.nn as nn

# Fully connected neural network with one hidden layer
class MLP(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate, max_epochs):
        '''
        input_size: [int], feature dimension 
        hidden_size: number of hidden nodes in the hidden layer
        output_size: number of classes in the dataset, 
        learning_rate: learning rate for gradient descent,
        max_epochs: maximum number of epochs to run gradient descent
        '''
        ### Construct your MLP Here (consider the recommmended functions in homework writeup)  
        super(MyMLP, self).__init__()

        # given arguments, see above
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

        # fully-connected layers
        self.fc1 = nn.Linear(input_size, hidden_size)  
        self.fc2 = nn.Linear(hidden_size, output_size) 
        # self.relu = nn.ReLU()

    def forward(self, x):
        ''' Function to do the forward pass with images x '''
        # pass input through fc1, apply ReLU, pass activated output through fc2
        # activation function given as nn.ReLU
        x = self.fc1(x)    # first layer
        x = nn.ReLU()(x)   # activation output
        return self.fc2(x) # return second layer out


    def fit(self, train_loader, criterion, optimizer):
        '''
        Function used to train the MLP

        train_loader: includes the feature matrix and class labels corresponding to the training set,
        criterion: the loss function used,
        optimizer: which optimization method to train the model.
        '''

        # lists for saving loss/error history
        losses = []
        error_rates = []
        
        # Epoch loop
        for i in range(self.max_epochs):
            # cumulative variables for loss/error
            loss_sum = 0.0
            correct = 0
            total = 0

            # Mini batch loop
            for j,(images,labels) in enumerate(train_loader):
                # Flatten images to 2D
                images = images.view(-1, self.input_size)

                # Forward pass (consider the recommmended functions in homework writeup)
                outputs = self.forward(images)
            
                # Backward pass and optimize (consider the recommmended functions in homework writeup)
                # Make sure to zero out the gradients using optimizer.zero_grad() in each loop
                optimizer.zero_grad()
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

        return losses, error_rates
        
    def predict(self, test_loader, criterion):
        '''
        Function used to predict with the MLP

        test_loader: includes the feature matrix and classlabels corresponding to the test set,
        criterion: the loss function used.
        '''
        # initialize counters
        loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad(): # no backprop step so turn off gradients
            for j,(images,labels) in enumerate(test_loader):
                # Compute prediction output and loss
                images = images.view(-1, self.input_size) # flatten again
                outputs = self.forward(images)

                # Track the loss and error rate
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1) # get index of predicted images
                total += labels.size(0)
                
                # count correct labels
                for i in range(len(predicted)):
                    if predicted[i] == labels[i]:
                        correct += 1

        # Print/return test loss and error rate
        loss = loss / len(test_loader)
        test_error_rate = 1 - correct / total
        print("Test Loss: %.4f | Test Error: %.4f" % (loss, test_error_rate))

        return loss, test_error_rate