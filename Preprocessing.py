import numpy as np
import os
import tensorflow
import matplotlib.pyplot as plt

def cropPatches(images, x1, x2, deltax1, deltax2, y1, y2, deltay1, deltay2): # Add black patch to image
    imgsCropPatch = np.zeros(images.shape, dtype = np.float32)
    for img in range(images.shape[0]):
        image = images[img]
        x_values = (x1, x2)
        y_values = (y1,y2)
        x = np.random.randint(x_values[0], x_values[1])
        delta_x = np.random.randint(deltax1, deltax2)
        y = np.random.randint(y_values[0], y_values[1])
        delta_y = np.random.randint(deltay1, deltay2)
        image[x:x+delta_x, y:y+delta_y] = 0
        imgsCropPatch[img] = image

    return imgsCropPatch

def loadImages(dataPath,imgSize, btchSize): # Load Images from dataset
    generator = tensorflow.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=False,
                                                                        vertical_flip=False)
    images = generator.flow_from_directory(dataPath, target_size=(imgSize, imgSize), color_mode='rgb', classes=None,
                                           class_mode=None, batch_size=btchSize, shuffle=True)
    return images
def scalingToOne(imgs): # Scaling images to [-1,1]
    imgs = (imgs - (255)/2) / (255/2)
    return tensorflow.constant(imgs)

def descaling(imgs): # Scaling images to [0,255]
    imgs = (imgs * (255)/2) + (255/2)
    imgs = tensorflow.clip_by_value(imgs, clip_value_min = 0, clip_value_max = 255)
    return imgs

def saveImgs(imgs, epoch): # Save generated images
    path = './savedImgs/'
    os.makedirs(path, exist_ok = True)
    for i in range(5):
        img = imgs[i]
        img = img.numpy().astype(np.uint8)
        #img = img.astype('uint8')
        plt.imsave(path + "image" + str(epoch) + "_" + str(i+1) + ".jpg", img)


### PREPROCESSING ###

imgSize = 128
btchSize = 64
trainDataset = loadImages('archive/tiny-imagenet-200/tiny-imagenet-200/train',imgSize,btchSize)
trainDatasetCrop = cropPatches(next(trainDataset), 40, 90, 15, 30, 40, 90, 15, 30)
trainDatasetCrop = scalingToOne(trainDatasetCrop)
trainDatasetCrop = descaling(trainDatasetCrop)

valDataset = loadImages('archive/tiny-imagenet-200/tiny-imagenet-200/val',imgSize,btchSize)
valDatasetCrop = cropPatches(next(valDataset), 40, 90, 15, 30, 40, 90, 15, 30)
valDatasetCrop = scalingToOne(valDatasetCrop)
valDatasetCrop = descaling(valDatasetCrop)

testDataset = loadImages('archive/tiny-imagenet-200/tiny-imagenet-200/test',imgSize,btchSize)
testDatasetCrop = cropPatches(next(testDataset), 40, 90, 15, 30, 40, 90, 15, 30)
testDatasetCrop = scalingToOne(testDatasetCrop)
testDatasetCrop = descaling(testDatasetCrop)

epochs = 1
saveImgs(trainDatasetCrop,epochs)
#saveImgs(valDatasetCrop,epochs)
#saveImgs(testDatasetCrop,epochs)


