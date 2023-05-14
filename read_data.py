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

trial = "patch_30_0.001"
filename = "./data_arrays/" + trial

data = np.load(filename)
nb_epochs = np.shape(data)
nb_epochs = nb_epochs[1]

plt.subplot(2, 2, 1)
plt.plot(np.linspace(1, nb_epochs, num=nb_epochs),  data[0, :], 'r', label= "Generator")
plt.plot(np.linspace(1, nb_epochs, num=nb_epochs), data[1, :], 'b', label= "Discriminator")
plt.xlabel("Epoch n°")
plt.ylabel("Loss function")
plt.legend()
plt.title("Value of the loss function for generator and discriminator (Training)")
plt.subplot(2, 2, 2)
plt.plot(np.linspace(1, nb_epochs, num=nb_epochs),  data[2, :], 'r', label= "Generator")
plt.plot(np.linspace(1, nb_epochs, num=nb_epochs), data[3, :], 'b', label= "Discriminator")
plt.xlabel("Epoch n°")
plt.ylabel("Loss function")
plt.legend()
plt.title("Value of the loss function for generator and discriminator (Validation)")
plt.subplot(2, 2, 3)
plt.plot(np.linspace(1, nb_epochs, num=nb_epochs),  data[4, :], 'r', label= "Generator")
plt.plot(np.linspace(1, nb_epochs, num=nb_epochs), data[5, :], 'b', label= "Discriminator")
plt.title("Accuracies of the discriminator (ability to detect false and true images) and the generator (ability to deceive the discriminator)")
plt.xlabel("Epoch n°")
plt.ylabel("Accuracy")
plt.legend()
plt.subplot(2, 2, 4)
plt.plot(np.linspace(1, nb_epochs, num=nb_epochs),  data[6, :], 'r', label= "Generator")
plt.plot(np.linspace(1, nb_epochs, num=nb_epochs), data[7, :], 'b', label= "Discriminator")
plt.xlabel("Epoch n°")
plt.ylabel("Mean absolute erorr")
plt.legend()
plt.show()

plt.plot(np.linspace(1, nb_epochs, num=nb_epochs), data[8, :], 'r', label='Generator')
plt.title('SSIM of the generated patch w.r.t. the original one')
plt.legend()
plt.show()