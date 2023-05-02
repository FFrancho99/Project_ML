# Training discriminator

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


def imshow(img):
    img = img / 2 + 0.5  # unnormalize to show images correctly
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def compute_run_acc(logits, labels):
    _, pred = torch.max(logits.data, 1)
    return (pred == labels).sum().item()


# Code from https://www.kaggle.com/code/yomnamabdulwahab/imagenet-pytorch
def load_dataset(data_path, batch_size):
    # Load all the images
    transformation = transforms.Compose([
        # Randomly augment the image data
        # Random horizontal flip
        # transforms.RandomHorizontalFlip(0.5),
        # Random vertical flip
        # transforms.RandomVerticalFlip(0.3),
        # crop image to a square image
        transforms.RandomCrop((128, 128)),
        # transform to tensors
        transforms.ToTensor(),
        # Normalize the pixel values (in R, G, and B channels)
        transforms.Normalize(mean=0.5, std=0.5)
    ])

    # Load all images, transforming them
    full_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=transformation
    )

    # Split into training (70% and testing (30%) datasets)
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.2 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    # use torch.utils.data.random_split for training/test split
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_dataset,
                                                                             [train_size, val_size, test_size])

    # define a loader for the training data we can iterate through in 50-image batches
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True
    )

    # define a loader for the testing data we can iterate through in 50-image batches
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False
    )

    return train_loader, val_loader, test_loader