from typing import Any

import torchvision.datasets

from Model import Generator
from Model import Discriminator
from TrainingFunctions import imshow
import TrainingFunctions
import os
import numpy as np
import matplotlib.pyplot as plt
import uuid
import torch
import torch.optim as optim
from torch import nn
import torchmetrics
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
from torchvision import transforms
from Preprocessing import *

mode = 1
train_folder = "dataset/afhq"
trial = 'test'
batch_size = 64

device = 'cuda:0'
generator_options = Generator.GeneratorOptions
generator = Generator.Generator(generator_options).to(device)

if mode == 0:
    discriminator = Discriminator.Discriminator(input_channels=3, size=64).to(device)
else:
    discriminator = Discriminator.Discriminator(input_channels=3, size=128).to(device)

discriminator.load_state_dict(torch.load('./Trained models/discriminator_'+trial+'.pth'))
generator.load_state_dict(torch.load('./Trained models/generator_'+trial+'.pth'))

_, _, test_loader = TrainingFunctions.load_dataset(train_folder, batch_size)

testiter = iter(test_loader)
real_batch, example_labels = next(testiter)
batch_im_crop, batch_patch_ori = cropPatches(real_batch, 64, 64)
predicted_patch = generator(batch_im_crop.to(device))

# Show images
imshow(torchvision.utils.make_grid(real_batch.cpu()))
predicted_image = merge_patch_image(predicted_patch, batch_im_crop, 64, 64)
imshow(torchvision.utils.make_grid(predicted_image.cpu().view(predicted_image.shape)))