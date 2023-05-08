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
#from ignite.metrics import SSIM
#from ignite.engine import Engine

# ### PREPROCESSING ###

### Load dataset ###
batch_size = 64
train_folder = "dataset/afhq"
trial = "test_100epochs"
nb_epochs = 100
learningRate = 0.0001

# Get the iterative dataloaders for test and training data
train_loader, val_loader, test_loader = TrainingFunctions.load_dataset(train_folder, batch_size)
print("Data loaders ready to read", train_folder)

dataiter = iter(train_loader)
example_images, _ = next(dataiter)
print(example_images.shape)
#TrainingFunctions.imshow(torchvision.utils.make_grid(example_images))

# Instanciate model

device = 'cuda:0'
generator_options = Generator.GeneratorOptions
generator = Generator.Generator(generator_options).to(device)
mode = 1
if mode == 0:
    discriminator = Discriminator.Discriminator(input_channels=3, size=64).to(device)
else:
    discriminator = Discriminator.Discriminator(input_channels=3, size=128).to(device)

real_label = 1
fake_label = 0

### Instanciate optimizer  ###

beta1 = 0.5


optimizerD = optim.Adam(discriminator.parameters(), lr=learningRate, betas=(beta1, 0.999))
optimizerG = optim.Adam(generator.parameters(), lr=learningRate, betas=(beta1, 0.999))

### Training ###
tr_lossesG = np.zeros(nb_epochs)
tr_lossesD = np.zeros(nb_epochs)
val_lossesG = np.zeros(nb_epochs)
val_lossesD = np.zeros(nb_epochs)
val_accuraciesG = np.zeros(nb_epochs)
val_accuraciesD = np.zeros(nb_epochs)
val_maesG = np.zeros(nb_epochs)
val_maesD = np.zeros(nb_epochs)



def criterionD_tr(predicted_proba, true_labels):
    criterion_adv = nn.BCELoss()  # adversarial criterion -> compare proba with label (Binary Cross Entropy loss)
    lossD = criterion_adv(predicted_proba, true_labels)
    return lossD


def criterionG_tr(predicted_proba, true_labels, predicted_patch, true_patch):
    weight_loss_adv = 0.001
    criterion_adv = nn.BCELoss()  # adversarial criterion -> compare proba with label (Binary Cross Entropy loss)
    criterion_rec = nn.MSELoss()  # reconstruction criterion -> compares patches (L2 loss)
    loss_adv = criterion_adv(predicted_proba, true_labels)
    loss_rec = criterion_rec(predicted_patch, true_patch)
    lossG = weight_loss_adv * loss_adv + (1 - weight_loss_adv) * loss_rec
    return lossG


def criterionD_val(predicted_proba, true_labels):
    criterion_MAE = nn.L1Loss()
    criterion_accuracy = torchmetrics.classification.BinaryAccuracy().to(device)
    lossD = criterionD_tr(predicted_proba, true_labels)
    maeD = criterion_MAE(predicted_proba, true_labels)
    accD = criterion_accuracy(predicted_proba, true_labels)
    return lossD, maeD, accD


def criterionG_val(predicted_proba, true_labels, predicted_patch, true_patch):
    criterion_MAE = nn.L1Loss()
    """metrics_SSIM = ignite.metrics.SSIM(data_range=255)
    metric_SSIM.attach(default_evaluator, 'ssim')
    state = default_evaluator.run([[predicted_patch, true_patch]])"""

    criterion_accuracy = torchmetrics.classification.BinaryAccuracy().to(device)
    lossG = criterionG_tr(predicted_proba, true_labels, predicted_patch, true_patch)
    maeG = criterion_MAE(predicted_proba, true_labels)
    accG = criterion_accuracy(predicted_proba, true_labels)
    return lossG, maeG, accG


for epoch_nr in range(nb_epochs):
    print("Epoch {}:".format(epoch_nr))
    # Train model
    running_lossD = 0.0
    running_lossG = 0.0
    running_lossD_val = 0.0
    running_lossG_val = 0.0
    running_accD_val = 0.0
    running_accG_val = 0.0
    running_maeD_val = 0.0
    running_maeG_val = 0.0
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
        if mode == 0:
            predicted_proba = discriminator(batch_patch_ori)
        else:
            predicted_proba = discriminator(batch_im_ori)
        lossD_real = criterionD_tr(predicted_proba, batch_labels)
        # compute gradient

        lossD_real.backward(retain_graph=True)

        ## train with fake input -> LABEL=0
        batch_labels.fill_(fake_label)

        # Predict and get loss
        predicted_patch = generator(batch_im_crop)
        if mode == 0:
            predicted_proba = discriminator(predicted_patch)
        else:
            predicted_image = merge_patch_image(predicted_patch, batch_im_crop, 64, 64)
            predicted_proba = discriminator(predicted_image)
        lossD_fake = criterionD_tr(predicted_proba, batch_labels)
        # compute gradient
        lossD_fake.backward(retain_graph=True)

        # update D
        lossD = (lossD_real + lossD_fake)/2.0
        optimizerD.step()

        ############################################
        ############# Train Generator ##############
        ############################################
        optimizerG.zero_grad()  # re-initialize the gradient to zero
        # Predict and get loss
        batch_labels.fill_(real_label)
        if mode == 0:
            predicted_proba = discriminator(predicted_patch)  # recompute the proba since D has been updated
        else:
            predicted_proba = discriminator(predicted_image)
        lossG = criterionG_tr(predicted_proba, batch_labels, predicted_patch, batch_patch_ori)

        # Update model

        lossG.backward()
        optimizerG.step()

        # Keep running statistics (for each batch)
        running_lossD += lossD.item()
        running_lossG += lossG.item()

    # Print results (for each epoch)
    tr_lossG = running_lossG / len(train_loader.dataset)
    tr_lossD = running_lossD / len(train_loader.dataset)
    print('>> TRAIN: Epoch {} completed | tr_lossG: {:.4f} | tr_lossD: {:.4f}'.format(
        epoch_nr, tr_lossG, tr_lossD))

    tr_lossesG[epoch_nr] = tr_lossG
    tr_lossesD[epoch_nr] = tr_lossD


    ############################################
    ############# Validation #############
    ############################################
    with torch.no_grad():
        for batch_im_ori_val, _ in val_loader:
            batch_im_crop_val, batch_patch_ori_val = cropPatches(batch_im_ori_val, 64, 64)
            # Put data on device
            batch_im_ori_val = batch_im_ori_val.to(device)
            batch_im_crop_val = batch_im_crop_val.to(device)
            batch_patch_ori_val = batch_patch_ori_val.to(device)

            # VALIDATION DISCRIMINATOR -- REAL INPUT -> LABEL=1
            batch_labels_val = torch.full((batch_im_ori_val.size(0), 1), real_label, dtype=torch.float,
                                          device=device)

            if mode == 0:
                predicted_proba_val = discriminator(batch_patch_ori_val)
            else:
                predicted_proba_val = discriminator(batch_im_ori_val)
            lossD_val_real, maeD_val_real, accD_val_real = criterionD_val(predicted_proba_val, batch_labels_val)

            # VALIDATION DISCRIMINATOR -- FAKE INPUT -> LABEL=0
            batch_labels_val.fill_(fake_label)
            predicted_patch_val = generator(batch_im_crop_val)
            if mode == 0:
                predicted_proba_val = discriminator(predicted_patch_val)
            else:
                predicted_image_val = merge_patch_image(predicted_patch_val, batch_im_crop_val, 64, 64)
                predicted_proba_val = discriminator(predicted_image_val)
            lossD_val_fake, maeD_val_fake, accD_val_fake = criterionD_val(predicted_proba_val, batch_labels_val)
            lossD_val = (lossD_val_real + lossD_val_fake) / 2.0
            maeD_val = (maeD_val_real + maeD_val_fake) / 2.0
            accD_val = (accD_val_real + accD_val_fake) / 2.0

            # VALIDATION GENERATOR -- FAKE INPUT but LABEL=1
            batch_labels_val.fill_(real_label)
            if mode == 0:
                predicted_proba_val = discriminator(predicted_patch_val)
            else:
                predicted_image_val = merge_patch_image(predicted_patch_val, batch_im_crop_val, 64, 64)
                predicted_proba_val = discriminator(predicted_image_val)

            lossG_val, maeG_val, accG_val = criterionG_val(predicted_proba_val, batch_labels_val, predicted_patch_val, batch_patch_ori_val)  # batch_data is the label here

            # Keep running statistics (for each batch)
            running_lossG_val += lossG_val
            running_lossD_val += lossD_val
            running_accG_val += accG_val
            running_accD_val += accD_val
            running_maeG_val += maeG_val
            running_maeD_val += maeD_val
    # at each epoch -> validation
    val_lossG = running_lossG_val / len(val_loader.dataset)
    print('>> VALIDATION: Epoch {} | val_loss: {:.4f}'.format(epoch_nr, val_lossG))
    val_lossesG[epoch_nr] = val_lossG
    val_lossesD[epoch_nr] = running_lossD_val / len(val_loader.dataset)
    val_accuraciesD[epoch_nr] = running_accD_val/(len(val_loader.dataset)/batch_size)
    val_accuraciesG[epoch_nr] = running_accG_val/(len(val_loader.dataset)/batch_size)
    val_maesD[epoch_nr] = running_maeD_val/(len(val_loader.dataset)/batch_size)
    val_maesG[epoch_nr] = running_maeG_val/(len(val_loader.dataset)/batch_size)



print('Training finished')

############################################
############# Test #############
############################################

testiter = iter(test_loader)
real_batch, example_labels = next(testiter)
batch_im_crop, batch_patch_ori = cropPatches(real_batch, 64, 64)
predicted_patch = generator(batch_im_crop.to(device))

# Show images
imshow(torchvision.utils.make_grid(real_batch.cpu()))
predicted_image = merge_patch_image(predicted_patch, batch_im_crop, 64, 64)
imshow(torchvision.utils.make_grid(predicted_image.cpu().view(predicted_image.shape)))

plt.subplot(2, 2, 1)
plt.plot(np.linspace(1, nb_epochs, num=nb_epochs),  tr_lossesG, 'r', label= "Generator")
plt.plot(np.linspace(1, nb_epochs, num=nb_epochs), tr_lossesD, 'b', label= "Discriminator")
plt.xlabel("Epoch n°")
plt.ylabel("Loss function")
plt.legend()
plt.title("Value of the loss function for generator and discriminator (Training)")
plt.subplot(2, 2, 2)
plt.plot(np.linspace(1, nb_epochs, num=nb_epochs),  val_lossesG, 'r', label= "Generator")
plt.plot(np.linspace(1, nb_epochs, num=nb_epochs), val_lossesD, 'b', label= "Discriminator")
plt.xlabel("Epoch n°")
plt.ylabel("Loss function")
plt.legend()
plt.title("Value of the loss function for generator and discriminator (Validation)")
plt.subplot(2, 2, 3)
plt.plot(np.linspace(1, nb_epochs, num=nb_epochs),  val_accuraciesG, 'r', label= "Generator")
plt.plot(np.linspace(1, nb_epochs, num=nb_epochs), val_accuraciesD, 'b', label= "Discriminator")
plt.xlabel("Epoch n°")
plt.ylabel("Accuracy")
plt.legend()
plt.subplot(2, 2, 4)
plt.plot(np.linspace(1, nb_epochs, num=nb_epochs),  val_maesG, 'r', label= "Generator")
plt.plot(np.linspace(1, nb_epochs, num=nb_epochs), val_maesD, 'b', label= "Discriminator")
plt.xlabel("Epoch n°")
plt.ylabel("Mean absolute erorr")
plt.legend()
plt.show()


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

# print('Training finished')
# testiter = iter(test_loader)
# real_batch, example_labels = next(testiter)
# batch_im_crop, batch_patch_ori = cropPatches(real_batch, 64, 64)
# predicted_patch = generator(batch_im_crop.to(device))

# Show images
#imshow(torchvision.utils.make_grid(batch_patch_ori.cpu()))
#imshow(torchvision.utils.make_grid(predicted_patch.cpu().view(batch_patch_ori.shape)))

torch.save(discriminator.state_dict(), './Trained models/discriminator_'+trial+'.pth')
torch.save(generator.state_dict(), './Trained models/generator_'+trial+'.pth')
