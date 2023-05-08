import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import random

#test commit
class Discriminator(nn.Module):
    def __init__(self, input_channels, size):
        super().__init__()
        self.input_size= size
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=6, kernel_size=(5, 5), padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        )#6@16x16

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )#16@8x8

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32,
                          kernel_size=(5, 5), padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )#32@4x4

        if(size == 64 or size == 128):
            self.layer4 = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=64,
                          kernel_size=(5, 5), padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            )
            if(size == 128):
                self.layer5 = nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=128,
                              kernel_size=(5, 5), padding=2),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
                )


        self.FClayers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_size * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        #print(x.shape)
        logits = self.layer3(self.layer2(self.layer1(x)))
        if(self.input_size == 64 or self.input_size == 128):
            logits = self.layer4(logits)
            if(self.input_size == 128):
                logits = self.layer5(logits)
        logits = self.FClayers(logits)
        return logits

