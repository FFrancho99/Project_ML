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
import Preprocessing

### PREPROCESSING ###

imgSize = 128
btchSize = 64
trainDataset = loadImages('archive/tiny-imagenet-200/tiny-imagenet-200/train',imgSize,btchSize)
trainDatasetCrop = cropPatches(next(trainDataset), 40, 90, 15, 30)
trainDatasetCrop = scalingToOne(trainDatasetCrop)
trainDatasetCrop = descaling(trainDatasetCrop)

valDataset = loadImages('archive/tiny-imagenet-200/tiny-imagenet-200/val',imgSize,btchSize)
valDatasetCrop = cropPatches(next(valDataset), 40, 90, 15, 30)
valDatasetCrop = scalingToOne(valDatasetCrop)
valDatasetCrop = descaling(valDatasetCrop)

testDataset = loadImages('archive/tiny-imagenet-200/tiny-imagenet-200/test',imgSize,btchSize)
testDatasetCrop = cropPatches(next(testDataset), 40, 90, 15, 30)
testDatasetCrop = scalingToOne(testDatasetCrop)
testDatasetCrop = descaling(testDatasetCrop)


saveImgs(trainDatasetCrop, epochs)

epochs = 50
batchSize = 64
learningRate = 0.0001

model_options.nC = 3 #rgb
model_options.neF = 64
model_options.nbF = 4000
model_options.nF = 64

generator = Generator.Generator(model_options)
discriminator = Discriminator.Discriminator(3)



def imshow(img):
    img = img / 2 + 0.5     # unnormalize to show images correctly
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def compute_run_acc(logits, labels):
    _, pred = torch.max(logits.data, 1)
    return (pred == labels).sum().item()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

batch_size = 8
num_epochs = 10
patch_size = 12
device = 'cuda:0'

transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5), (0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
valset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

classes = ('True', 'False')

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                         shuffle=False)

learning_rate = 0.0002
beta1 = 0.5
discriminator = Discriminator(3).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, 0.999))

blur = torchvision.transforms.GaussianBlur(5)

for epoch_nbr in range(num_epochs):
    print("Epoch {}:".format(epoch_nbr))
    running_loss = 0.0
    for batch_data, _ in trainloader:

        batch_labels = torch.ones([batch_size, 1], dtype=torch.float)
        batch_labels = batch_labels.to(device)
        batch_data = batch_data.to(device)

        pred = discriminator(batch_data)
        loss = criterion(pred, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        ind1 = random.randrange(0, 31-patch_size)
        ind2 = random.randrange(0, 31 - patch_size)
        batch_labels = torch.zeros([batch_size, 1], dtype=torch.float).to(device)

        for i in range(batch_size):
            batch_data = batch_data.to('cpu')
            image = batch_data[i, :, :, :]
            image = blur(image)
            batch_data[i, :, ind1:ind1+patch_size-1, ind2:ind2+patch_size-1] = image[:, ind1:ind1+patch_size-1,
                                                                                ind2:ind2+patch_size-1]

        batch_data = batch_data.to(device)
        pred = discriminator(batch_data)
        loss = criterion(pred, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print('>> TRAIN: Epoch {} completed | tr_loss: {:.4f}'.format(
        epoch_nbr, running_loss / (2*len(trainloader.dataset))))

    running_loss = 0.0
    running_acc = 0.0

    #VALIDATION
    for batch_data, _ in valloader:
        batch_data = batch_data.to(device)
        batch_labels = torch.ones([batch_size, 1], dtype=torch.float)
        batch_labels = batch_labels.to(device)

        pred = discriminator(batch_data)
        pred = torch.round(pred)
        running_acc += sum(pred == batch_labels).item()

        ind1 = random.randrange(0, 31 - patch_size)
        ind2 = random.randrange(0, 31 - patch_size)
        batch_labels = torch.zeros([batch_size, 1], dtype=torch.float).to(device)

        for i in range(batch_size):
            batch_data = batch_data.to('cpu')
            image = batch_data[i, :, :, :]
            image = blur(image)
            batch_data[i, :, ind1:ind1 + patch_size - 1, ind2:ind2 + patch_size - 1] = image[:,
                                                                                       ind1:ind1 + patch_size - 1,
                                                                                       ind2:ind2 + patch_size - 1]

        batch_data = batch_data.to(device)
        pred = discriminator(batch_data)
        pred = torch.round(pred)
        running_acc += sum(pred == batch_labels).item()

    val_acc = 100*running_acc /( 2*len(valloader.dataset))
    print('>> VALIDATION: Epoch {} | val_acc: {:.2f}%'.format(epoch_nbr, val_acc))