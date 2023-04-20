import os
import numpy as np
import matplotlib.pyplot as plt
import uuid
import torch
from torch import nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(4, 4), stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(4, 4), stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

    def forwardPropagation(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

    print('Adeline est cool ou pas?')
