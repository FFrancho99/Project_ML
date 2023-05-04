import numpy as np
import os
import tensorflow
import torch
import matplotlib.pyplot as plt


def cropPatches(images, x, deltax):  # Add black patch to image

    x1 = x - (deltax // 2)
    x2 = x + (deltax // 2)
    imgsCropPatch = images
    patches = images[:, :, x1:x2, x1:x2]
    imgsCropPatch[:, :, x1:x2, x1:x2] = 0

    return imgsCropPatch, patches

def merge_patch_image(patches, images, x, deltax):
    x1 = x - (deltax // 2)
    x2 = x + (deltax // 2)
    output = torch.clone(images)
    output[:, :, x1:x2, x1:x2] = torch.clone(patches)

    return output


def loadImages(dataPath, imgSize, btchSize):  # Load Images from dataset
    generator = tensorflow.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=False,
                                                                        vertical_flip=False)
    images = generator.flow_from_directory(dataPath, target_size=(imgSize, imgSize), color_mode='rgb', classes=None,
                                           class_mode=None, batch_size=btchSize, shuffle=True)
    return images


def scalingToOne(imgs):  # Scaling images to [-1,1]
    imgs = (imgs - (255) / 2) / (255 / 2)
    return tensorflow.constant(imgs)


def descaling(imgs):  # Scaling images to [0,255]
    imgs = (imgs * (255) / 2) + (255 / 2)
    imgs = tensorflow.clip_by_value(imgs, clip_value_min=0, clip_value_max=255)
    return imgs


def saveImgs(imgs, epoch):  # Save generated images
    path = './savedImgs/'
    os.makedirs(path, exist_ok=True)
    for i in range(5):
        img = imgs[i]
        img = img.numpy().astype(np.uint8)
        # img = img.astype('uint8')
        plt.imsave(path + "image" + str(epoch) + "_" + str(i + 1) + ".jpg", img)
