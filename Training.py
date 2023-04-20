from Model import Generator
from Model import Discriminator
import os
import numpy as np
import matplotlib.pyplot as plt
import uuid
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

epochs = 50
batchSize = 64
learningRate = 0.0001

generator = Generator.Generator()

discriminator = Discriminator.Discriminator()