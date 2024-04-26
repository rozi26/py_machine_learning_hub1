import sys
sys.path.append("F:\\programing\\python\\machine_learning_hub")
import matplotlib.pylab as plt

from cv2 import imwrite
import numpy as np

#from Models.GAN_MC.Generator import Generator

def showImage(img, text=None):
    getImage(img,text)
    plt.show()
def getImage(img, text=None):
    plt.close()
    plt.axis("off")
    if(len(img[0][0]) == 1):
        plt.imshow(img,cmap="gray")
    else:
        plt.imshow((img))
    if(text != None): plt.xlabel(text)

def saveImage(img, file):
    plt.close()
    plt.imshow(img)
    plt.savefig(file)


def ganerateImages(generator, size, path):
    plt.close()
    fig, axes = plt.subplots(size[0],size[1],figsize=(15,15))
    fig.tight_layout(pad=1.0)
    for a1 in range(size[0]):
        for a2 in range(size[1]):
            predict = generator.generate()[0]
            if(len(predict[0][0]) == 1):
                axes[a1,a2].imshow(predict,cmap='gray')
            else:
                axes[a1,a2].imshow(predict)
    plt.savefig(path)

def showImages(images, shape):
    images = np.reshape(images, shape + images.shape[1:])
    getImages(images)
    plt.show()

def saveImages(images,path):
    getImages(images)
    plt.savefig(path)

def getImages(images):
    plt.close()
    fig, axes = plt.subplots(len(images),len(images[0]),figsize=(15,15))
    for a1 in range(len(images)):
        for a2 in range(len(images[0])):
            if(len(images[0][0]) == 1):
                axes[a1,a2].imshow(images[a1][a2],cmap='gray')
            else:
                axes[a1,a2].imshow(images[a1][a2])
