import numpy as np
from PIL import Image
import torchvision

def crop_patches(images):
    imgCropPatch = np.zeros(images.shape, dtype = np.float32)
    for img in range(images.shape[0]):
        image = images[img]
        x_values = (40, 90)
        y_values = (40,90)
        x = np.random.randint(x_values[0], x_values[1])
        delta_x = np.random.randint(15, 30)
        y = np.random.randint(y_values[0], y_values[1])
        delta_y = np.random.randint(15, 30)
        image[x:x+delta_x, y:y+delta_y] = 0
        imgCropPatch[img] = image

    return imgCropPatch

def load_imges_TinyImaneNet():
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(size=256, interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
    ])
    train_dataset = torchvision.datasets.ImageFolder(
        root='/kaggle/input/tiny-imagenet/tiny-imagenet-200/train',
        transform=transforms,
    )
    val_dataset = torchvision.datasets.ImageFolder(
        root='/kaggle/input/tiny-imagenet/tiny-imagenet-200/val',
        transform=transforms,
    )
    test_dataset = torchvision.datasets.ImageFolder(
        root='/kaggle/input/tiny-imagenet/tiny-imagenet-200/test',
        transform=transforms,
    )
    return train_dataset,val_dataset,test_dataset
