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
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
from torchvision import transforms
from Preprocessing import *

# ### PREPROCESSING ###
#
# imgSize = 128

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
batch_size = 64
train_folder = "dataset/imagewoof2-160"
nb_epochs = 10
learningRate = 0.0001

# Get the iterative dataloaders for test and training data
train_loader, val_loader, test_loader = TrainingFunctions.load_dataset(train_folder, batch_size)
print("Data loaders ready to read", train_folder)

dataiter = iter(train_loader)
example_images, _ = next(dataiter)
print(example_images.shape)
TrainingFunctions.imshow(torchvision.utils.make_grid(example_images))

# Instanciate model
device = 'cuda:0'
generator_options = Generator.GeneratorOptions
generator = Generator.Generator(generator_options).to(device)
discriminator = Discriminator.Discriminator(input_channels=3, size=64).to(device)
real_label = 1
fake_label = 0

### Instanciate optimizer  ###

beta1 = 0.5
criterion = nn.BCELoss()
optimizerD = optim.Adam(discriminator.parameters(), lr=learningRate, betas=(beta1, 0.999))
optimizerG = optim.Adam(generator.parameters(), lr=learningRate, betas=(beta1, 0.999))

### Training ###
tr_lossesG = np.zeros(nb_epochs)
tr_lossesD = np.zeros(nb_epochs)
val_lossesG = np.zeros(nb_epochs)
val_lossesD = np.zeros(nb_epochs)


for epoch_nr in range(nb_epochs):

    print("Epoch {}:".format(epoch_nr))
    # Train model
    running_lossD = 0.0
    running_lossG = 0.0
    for batch_im_ori, _ in train_loader:
        ############################################
        ########### Train Discriminator ############
        ############################################
        optimizerD.zero_grad()

        # remove (crop) patch from image
        batch_im_crop, batch_patch_ori = cropPatches(batch_im_ori, 64, 64)
        # Put data on device
        batch_im_ori = batch_im_ori.to(device)
        batch_im_crop = batch_im_crop.to(device)
        batch_patch_ori = batch_patch_ori.to(device)

        ## train with real input -> LABEL=1
        batch_labels = torch.full((batch_im_ori.size(0), 1), real_label, dtype=torch.float, device=device)

        # Predict and get loss
        predicted_proba = discriminator(batch_patch_ori)
        lossD_real = criterion(predicted_proba, batch_labels)  # batch_data = label here
        # compute gradient

        lossD_real.backward(retain_graph=True)

        ## train with fake input -> LABEL=0
        batch_labels.fill_(fake_label)

        # Predict and get loss
        predicted_patch = generator(batch_im_crop)
        predicted_proba = discriminator(predicted_patch)
        lossD_fake = criterion(predicted_proba, batch_labels)  # batch_data = label here
        # compute gradient
        lossD_fake.backward(retain_graph=True)

        # update D
        lossD = lossD_real + lossD_fake
        optimizerD.step()

        ############################################
        ############# Train Generator ##############
        ############################################
        optimizerG.zero_grad() # re-initialize the gradient to zero
        # Predict and get loss
        batch_labels.fill_(real_label)
        predicted_proba = discriminator(predicted_patch) # recompute the proba since D has been updated
        lossG = criterion(predicted_proba, batch_labels)  # batch_data = label here

        # Update model

        lossG.backward()
        optimizerG.step()

        # Keep running statistics
        running_lossD += lossD.item()
        running_lossG += lossG.item()

    # Print results
    tr_lossG = running_lossG / len(train_loader.dataset)
    tr_lossD = running_lossD / len(train_loader.dataset)
    print('>> TRAIN: Epoch {} completed | tr_lossG: {:.4f} | tr_lossD: {:.4f}'.format(
        epoch_nr, tr_lossG, tr_lossD))

    tr_lossesG[epoch_nr] = tr_lossG
    tr_lossesD[epoch_nr] = tr_lossD
    # Get validation results
    running_loss = 0

    # with torch.no_grad():
    #     for batch_im_ori, batch_im_crop in val_loader:
    #         # Put data on device
    #         batch_im_ori = batch_im_ori.to(device)
    #         batch_im_crop = batch_im_crop.to(device)
    #
    #         # Predict and get loss
    #         predicted_patch = generator(batch_im_crop)
    #         loss = criterion(predicted_patch, batch_im_ori)  # batch_data is the label here
    #
    #         # Keep running statistics
    #         running_loss += criterion(predicted_patch, batch_im_ori)  # batch_data = label
    #
    # val_loss = running_loss / len(val_loader.dataset)
    # print('>> VALIDATION: Epoch {} | val_loss: {:.4f}'.format(epoch_nr, val_loss))
    #
    #
    # val_lossesG[epoch_nr] = val_loss

print('Training finished')
testiter = iter(test_loader)
real_batch, example_labels = next(testiter)
batch_im_crop, batch_patch_ori = cropPatches(real_batch, 64, 64)
predicted_patch = generator(batch_im_crop.to(device))


# Show images
imshow(torchvision.utils.make_grid(batch_patch_ori.cpu()))
imshow(torchvision.utils.make_grid(predicted_patch.cpu().view(batch_patch_ori.shape)))

