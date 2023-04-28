import os
import numpy as np
import matplotlib.pyplot as plt
import uuid
import torch
from torch import nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms


class GeneratorOptions:
    nC = 3  # rgb
    neF = 64  # nb encoder features
    nbF = 1000  # nb bottleneck features
    ndF = 64  # nb decoder features

    def __init__(self):
        self.nC = 3  # rgb
        self.neF = 64  # nb encoder features
        self.nbF = 1000  # nb bottleneck features
        self.ndF = 64  # nb decoder features


class Generator(nn.Module):
    def __init__(self, mod_opt):
        self.mod_opt = mod_opt
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            # encoder: conv+AF+pool (pool done with the stride 2 in conv)
            # 128x128 x nC
            nn.Conv2d(mod_opt.nC, mod_opt.neF, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 64x64 x neF
            nn.Conv2d(mod_opt.neF, mod_opt.neF*2, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 32x32 x neF
            nn.Conv2d(mod_opt.neF*2, mod_opt.neF * 4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 16x16 x neF*2
            nn.Conv2d(mod_opt.neF * 4, mod_opt.neF * 8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 8x8 x neF*4
            nn.Conv2d(mod_opt.neF * 8, mod_opt.neF * 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=4, stride=4),
            #nn.Conv2d(mod_opt.neF * 4, mod_opt.neF * 8, 4, 2, 0),
            #nn.BatchNorm2d(mod_opt.neF * 8),
            #nn.ReLU(True)
            # 4x4 x neF*8

            #nn.Conv2d(mod_opt.neF * 8, mod_opt.nbF, 4, 1, 0),
            #nn.BatchNorm2d(mod_opt.nbF),
            #nn.ReLU(True)
            # 1x1 x nbF
        )


        self.decoder = nn.Sequential(
            # 1x1 x nbF
            # 8x8 x ndF*4
            nn.ConvTranspose2d(mod_opt.ndF * 16, mod_opt.ndF * 8, kernel_size=5, stride=4, padding=1),
            nn.ReLU(True),
            # 16x16 x ndF*2
            nn.ConvTranspose2d(mod_opt.ndF * 8, mod_opt.ndF*4, kernel_size=5, stride=2, padding=1),
            nn.ReLU(True),
            # 32x32 x ndF
            nn.ConvTranspose2d(mod_opt.ndF*4, mod_opt.ndF*2, kernel_size=5, stride=2, padding=1),
            nn.ReLU(True),
            # 64x64 x ndF

            nn.ConvTranspose2d(mod_opt.ndF * 2, mod_opt.nC, kernel_size=4, stride=2, padding=0),

            #nn.ConvTranspose2d(mod_opt.ndF * 2, mod_opt.ndF, kernel_size=5, stride=2, padding=1),
            #nn.ReLU(True),
            #nn.ConvTranspose2d(mod_opt.ndF, mod_opt.nC, kernel_size=4, stride=2, padding=0),
            # 64x64 x nc
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.tanh(x)
        return x
