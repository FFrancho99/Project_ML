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
from torchmetrics.functional import structural_similarity_index_measure
import torchmetrics
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
from torchvision import transforms
from Preprocessing import *

#1 figure: SSIM for 30 epochs, 3 curves: weight_loss_adv (0.001 0.01 0.1 0.5)

mode = "full"
trial0_1 = mode+ "_30_0.1"
trial0_01 =mode+ "_30_0.01"
trial0_001 = mode+"_30_0.001"
filename = "./data_arrays/" #+ trial

path = (filename + trial0_1, filename + trial0_01, filename + trial0_001)

data0_1 = np.load(path[0])
data0_01 = np.load(path[1])
data0_001 = np.load(path[2])
nb_epochs = 30

x_axis = np.linspace(1, nb_epochs, num=nb_epochs)
plt.plot(x_axis, data0_1[8, :], 'b', label='weight loss adv = 0.1')
plt.plot(x_axis, data0_01[8, :], 'r', label='weight loss adv = 0.01')
plt.plot(x_axis, data0_001[8, :], 'g', label='weight loss adv = 0.001')
plt.title('SSIM for different values of the weight loss adv')
plt.xlabel("Epoch n°")
plt.ylabel("SSIM")
plt.legend()
plt.show()

#1 figure: SSIM for 30 epochs, large weight_loss_adv: 2 curves: whole image vs patch

data_patch = np.load(filename + "patch_30_0.001")

plt.plot(x_axis, data0_001[8, :], 'g', label='Full')
plt.plot(x_axis, data_patch[8, :], 'r', label='Patch')
plt.title('SSIM for full image vs patch only')
plt.xlabel("Epoch n°")
plt.ylabel("SSIM")
plt.legend()
plt.show()

#1 figure: loss fct for 30 epochs, 2 curves: G and D



#1 figure avec beaux resultats (tell for which settings)

mode = 0
train_folder = "dataset/afhq"
trial = 'patch_30_0.001'
batch_size = 32

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