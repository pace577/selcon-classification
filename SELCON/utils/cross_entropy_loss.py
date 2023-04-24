# Write a function to take a model, a dataset and evaluate a cross entropy loss on the dataset.
# Path: SELCON\utils\cross_entropy_loss.py

'''
`Use this file to evaluate your subset`

Usage::
    loss_trn, loss_val = cross_entropy_loss(x_trn, y_trn, x_sub, y_sub, x_val, y_val)

'''

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import matplotlib.pyplot as plt
import pdb

sys.path.append(os.path.join(".","SELCON"))
sys.path.append(os.path.join(".","SELCON\\utils"))
sys.path.append(os.path.join(".","SELCON\\utils\\custom_dataset"))
sys.path.append(os.path.join(".","SELCON\\model"))
sys.path.append(os.path.join(".","SELCON\\gen_result"))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the logistic regression model
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


def train_logistic_regression(X, y, input_dim, lr=0.01, epochs=1, verbose = False):
    # Define the logistic regression model
    model = LogisticRegression(input_dim)
    model = model.to(device)
    X = X.to(device)
    y = y.to(device)
    # Initialize the loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Train the model for the specified number of epochs
    for epoch in range(epochs):
        # Forward pass
        y_pred = model(X)
        # Compute the loss
        loss = criterion(y_pred, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose:
            # Print the loss
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

    # Return the trained model
    return model


def cross_entropy_loss(x_trn, y_trn, x_sub, y_sub, x_val, y_val, epochs=100, lr = 0.05):
    # train a model on x_trn, y_trn and evaluate the cross entropy loss on x_val, y_val
    criterion = nn.BCELoss()

    x_trn = x_trn.to(device)
    y_trn = y_trn.to(device)
    x_val = x_val.to(device)
    y_val = y_val.to(device)
    x_sub = x_sub.to(device)
    y_sub = y_sub.to(device)

    if len(y_trn.shape) == 1:
            y_trn = y_trn.unsqueeze(1)
    if len(y_sub.shape) == 1:
            y_sub = y_sub.unsqueeze(1)
    if len(y_val.shape) == 1:
            y_val = y_val.unsqueeze(1)
    
    N, M = x_trn.shape
    # Train a LR model on x_trn, y_trn and evaluate its loss on the validation set
    trnmodel = train_logistic_regression(x_trn, y_trn, input_dim=M, lr=lr, epochs=epochs)
    y_val_trnmodel = trnmodel(x_val)
    loss_val_trnmodel = criterion(y_val_trnmodel, y_val)

    N, M = x_sub.shape
    # Train a LR model on x_sub, y_sub and evaluate its loss on the validation set
    submodel = train_logistic_regression(x_sub, y_sub, input_dim=M, lr=lr, epochs=epochs)
    y_val_submodel = submodel(x_val)
    loss_val_submodel = criterion(y_val_submodel, y_val)
    
    # print(f"loss_val_trnmodel: {loss_val_trnmodel.item():.4f}")
    # print(f"loss_val_submodel: {loss_val_submodel.item():.4f}")
    
    return loss_val_trnmodel, loss_val_submodel

if __name__ == '__main__':
    # Define the dataset and labels
    X = torch.randn(100, 10) # input data of size (batch_size, M)
    y = torch.randint(0, 2, (100, 1)).float() # binary labels of size (batch_size, 1)
    X = X.to(device)
    y = y.to(device)
    
    # Train the logistic regression model
    model = train_logistic_regression(X, y, input_dim=10, lr=0.01, epochs=100, verbose=True)
    
    criterion = nn.BCELoss()
    y_pred = model(X)
    loss = criterion(y_pred, y)

    print(f'Loss: {loss.item():.4f}')

    trn_loss, sub_loss = cross_entropy_loss(X, y, X, y, X, y)