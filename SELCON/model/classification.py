#!/usr/bin/env python3

import torch
import torch.nn as nn

class LogisticRegression(nn.Module):
    def __init__(self,input_dim,n_classes):
        super(LogisticRegression,self).__init__()
        self.linear = nn.Linear(input_dim,1)

    def forward(self,x):
        scores = self.linear(x)
        scores = torch.sigmoid(self.linear(x).view(-1))
        return scores
