import numpy as np
import os
import tensorflow
import matplotlib.pyplot as plt


def cropPatches(images, x, deltax):  # Add black patch to image
    imgsCropPatch = np.zeros(images.shape, dtype=np.float32)
    for img in range(images.shape[0]):
        image = images[img]
        image[x - deltax / 2:x + deltax / 2, x - deltax / 2:x + deltax / 2] = 0
        imgsCropPatch[img] = image
    return imgsCropPatch


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
