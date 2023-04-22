import os
import numpy as np
import matplotlib.pyplot as plt
import uuid
import torch
from torch import nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms


class Generator(nn.Module):
    def __init__(self, mod_opt):
        super(Generator, self).__init__()

        self.encoder = nn.Sequential(
            # encoder: conv+AF+pool (pool done with the stride 2 in conv)
            # 128x128 x nC
            nn.Conv2d(mod_opt.nC, mod_opt.neF, kernel_size=(4, 4), stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 64x64 x neF
            nn.Conv2d(mod_opt.neF, mod_opt.neF, 4, 2, 1),
            nn.BatchNorm2d(mod_opt.neF),
            nn.ReLU(True),
            # 32x32 x neF
            nn.Conv2d(mod_opt.neF, mod_opt.neF * 2, 4, 2, 1),
            nn.BatchNorm2d(mod_opt.neF * 2),
            nn.ReLU(True),
            # 16x16 x neF*2
            nn.Conv2d(mod_opt.neF * 2, mod_opt.neF * 4, 4, 2, 1),
            nn.BatchNorm2d(mod_opt.neF * 4),
            nn.ReLU(True),
            # 8x8 x neF*4
            nn.Conv2d(mod_opt.neF * 4, mod_opt.neF * 8, 4, 2, 1),
            nn.BatchNorm2d(mod_opt.neF * 8),
            nn.ReLU(True),
            # 4x4 x neF*8
            nn.Conv2d(mod_opt.neF * 8, mod_opt.nbF, 4, 1, 0),
            nn.BatchNorm2d(mod_opt.nbF),
            nn.ReLU(True)
            # 1x1 x nbF
        )

        self.decoder = nn.Sequential(
            # 1x1 x nbF
            nn.ConvTranspose2d(mod_opt.nbF, mod_opt.ndF * 8, 4, 1, 0),
            nn.BatchNorm2d(mod_opt.nbF*8),
            nn.ReLU(True),
            # 4x4 x ndF*8
            nn.ConvTranspose2d(mod_opt.ndF * 8, mod_opt.ndF * 4, 4, 2, 1),
            nn.BatchNorm2d(mod_opt.nbF*4),
            nn.ReLU(True),
            # 8x8 x ndF*4
            nn.ConvTranspose2d(mod_opt.ndF * 4, mod_opt.ndF * 2, 4, 2, 1),
            nn.BatchNorm2d(mod_opt.nbF*2),
            nn.ReLU(True),
            # 16x16 x ndF*2
            nn.ConvTranspose2d(mod_opt.ndF * 2, mod_opt.ndF, 4, 2, 1),
            nn.BatchNorm2d(mod_opt.nbF),
            nn.ReLU(True),
            # 32x32 x ndF
            nn.ConvTranspose2d(mod_opt.ndF, mod_opt.ndF, 4, 2, 1),
            nn.BatchNorm2d(mod_opt.nbF),
            nn.ReLU(True),
            # 64x64 x ndF
            nn.ConvTranspose2d(mod_opt.ndF, mod_opt.nc, 4, 2, 1),
            nn.BatchNorm2d(mod_opt.nbF)
            # 64x64 x nc
        )

    def forwardPropagation(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.tanh(x)
        return x
