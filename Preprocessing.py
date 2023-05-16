import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from TrainingFunctions import imshow
import torchvision.datasets


def cropPatches(images, x, deltax):  # Add black patch to image

    x1 = x - (deltax // 2)
    x2 = x + (deltax // 2)
    imgsCropPatch = torch.clone(images)
    patches = torch.clone(images[:, :, x1:x2, x1:x2])
    imgsCropPatch[:, :, x1:x2, x1:x2] = 0

    return imgsCropPatch, patches


def merge_patch_image(patches, images, x, deltax):
    x1 = x - (deltax // 2)
    x2 = x + (deltax // 2)
    output = torch.clone(images)
    output[:, :, x1:x2, x1:x2] = torch.clone(patches)

    return output
