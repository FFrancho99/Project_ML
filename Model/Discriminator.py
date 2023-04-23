import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import random

#test commit
class Discriminator(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=6,
                      kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )  # 6@14x14

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16,
                      kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )  # 16@5x5
        self.layer3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        logits = self.layer2(self.layer1(x))
        logits = self.layer4(self.layer3(logits))
        return logits


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
num_epochs = 20
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