import torchvision.datasets

from Model import Generator
from Model import Discriminator
import os
import numpy as np
import matplotlib.pyplot as plt
import uuid
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
from torchvision import transforms
from Preprocessing import *

# ### PREPROCESSING ###
#
# imgSize = 128
batch_size = 64
# trainDataset = loadImages('archive/tiny-imagenet-200/tiny-imagenet-200/train', imgSize, btchSize)
# trainLoader, trainLoaderCrop = cropPatches(trainDataset, 64, 64)
# trainLoaderCrop = scalingToOne(trainLoaderCrop)
# # trainLoaderCrop = descaling(trainLoaderCrop)
#
# valDataset = loadImages('archive/tiny-imagenet-200/tiny-imagenet-200/val', imgSize, btchSize)
# valLoader, valLoaderCrop = cropPatches(valDataset, 64, 64)
# valLoaderCrop = scalingToOne(valLoaderCrop)
# # valLoaderCrop = descaling(valLoaderCrop)
#
# testDataset = loadImages('archive/tiny-imagenet-200/tiny-imagenet-200/test', imgSize, btchSize)
# testLoader, testLoaderCrop = cropPatches(testDataset, 64, 64)
# testLoaderCrop = scalingToOne(testLoaderCrop)
# # testLoaderCrop = descaling(testLoaderCrop)
#
# saveImgs(trainLoaderCrop, 1)

### Load dataset ###
transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5), (0.5))])


trainset = torchvision.datasets.ImageFolder(root='archive/tiny-imagenet-200/tiny-imagenet-200/train', transform=transform)
valset = torchvision.datasets.ImageFolder(root='archive/tiny-imagenet-200/tiny-imagenet-200/val', transform=transform)

print(trainset)

# Create dataloaders
trainLoader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)
valLoader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                        shuffle=False)


### Instanciate model  ###

generator_options = Generator.GeneratorOptions
generator = Generator.Generator(generator_options)
discriminator = Discriminator.Discriminator(3)

### Instanciate optimizer  ###

nb_epochs = 1
batchSize = 64
learningRate = 0.0001

criterion = nn.BCELoss()
optimizer = optim.Adam(discriminator.parameters(), lr=learningRate)

### Training ###
device = 'cpu'
tr_losses = np.zeros(nb_epochs)
val_losses = np.zeros(nb_epochs)

for epoch_nr in range(nb_epochs):

    print("Epoch {}:".format(epoch_nr))

    # Train model
    running_loss = 0.0
    for batch_im_ori, _ in trainLoader:

        # remove (crop) patch from image
        batch_im_crop, batch_patch_ori = cropPatches(batch_im_ori, 32, 32)
        print(batch_im_crop.shape)
        print(batch_patch_ori.shape)
        # Put data on device
        batch_im_ori = batch_im_ori.to(device)
        batch_im_crop = batch_im_crop.to(device)

        # Predict and get loss
        predicted_patch = generator(batch_im_crop)
        print(predicted_patch.shape)
        loss = criterion(predicted_patch, batch_patch_ori)  # batch_data = label here

        # Update model
        optimizer.zero_grad()  # re-initialize the gradient to zero
        loss.backward()
        optimizer.step()

        # Keep running statistics
        running_loss += loss.item()

    # Print results
    tr_loss = running_loss / len(trainLoader.dataset)
    print('>> TRAIN: Epoch {} completed | tr_loss: {:.4f}'.format(
        epoch_nr, running_loss / len(trainLoader.dataset)))

    # Get validation results
    running_loss = 0

    with torch.no_grad():
        for batch_im_ori, batch_im_crop in valLoader:
            # Put data on device
            batch_im_ori = batch_im_ori.to(device)
            batch_im_crop = batch_im_crop.to(device)

            # Predict and get loss
            predicted_patch = generator(batch_im_crop)
            loss = criterion(predicted_patch, batch_im_ori)  # batch_data is the label here

            # Keep running statistics
            running_loss += criterion(predicted_patch, batch_im_ori)  # batch_data = label

    val_loss = running_loss / len(valLoader.dataset)
    print('>> VALIDATION: Epoch {} | val_loss: {:.4f}'.format(epoch_nr, val_loss))

    tr_losses[epoch_nr] = tr_loss
    val_losses[epoch_nr] = val_loss

print('Training finished')


def imshow(img):
    img = img / 2 + 0.5  # unnormalize to show images correctly
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def compute_run_acc(logits, labels):
    _, pred = torch.max(logits.data, 1)
    return (pred == labels).sum().item()
