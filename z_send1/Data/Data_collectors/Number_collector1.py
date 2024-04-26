import sys
sys.path.append("F:\\programing\\python\\machine_learning_hub")

from Models.GAN1_MC.GANS.GAN2 import GAN2
from Data.ImageFolders import ReadImageDiractry
from Graphics.ShowImages import showImage, ganerateImages
from Models.GAN1_MC.Generator import modelG_keras_28x28_1,Generator
from Models.GAN1_MC.Discriminator import model_keras_28x28_1,Discriminator
from Models.GAN1_MC.GANS.GAN1 import GAN
from Graphics.ProgressBar import ProgressBar
from keras.datasets.mnist import load_data
from keras.utils import to_categorical
import tensorflow as tf
import numpy as np

PATH1 = "F:\\storge\\Machine Learning Models\\numbers\\"

def getKerasMinst():
    (x_train, y_train), (x_test, y_test) = load_data()
    all_digits = np.concatenate([x_train, x_test])
    all_labels = np.concatenate([y_train, y_test])

    # Scale the pixel values to [0, 1] range, add a channel dimension to
    # the images, and one-hot encode the labels.
    all_digits = all_digits.astype("float32") / 255.0
    all_digits = np.reshape(all_digits, (-1, 28, 28, 1))
    all_labels = to_categorical(all_labels, 10)

    # Create tf.data.Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((all_digits, all_labels))
    dataset = dataset.shuffle(buffer_size=1024).batch(64)
    return dataset

CLASSES = 2
BUFFERS = 1
EPOCHES = 2
GENERATOR_INPUT = 100

def imagesToTF(data):
    labels = np.ones(len(data),dtype=np.float32)
    labels = to_categorical(labels, CLASSES)
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.shuffle(buffer_size=1024).batch(64)
    print(f"Shape of training images: {data.shape}")
    print(f"Shape of training labels: {labels.shape}")
    return dataset

data = ReadImageDiractry(PATH1,buffers=BUFFERS,image_size=(28,28),convetion=1)

model_d = model_keras_28x28_1()
model_g = modelG_keras_28x28_1(GENERATOR_INPUT + CLASSES)
#gan = GAN(model_g,model_d,generator_input_shape=GENERATOR_INPUT,discriminator_input_shape=(28,28),name="number generator")

gan = GAN2(model_d,model_g,GENERATOR_INPUT,28,name="z_gan2_number_generator")
gan.compile1()

images = data.loadBatch(0)
#dataset = imagesToTF(images)
#print(dataset)
#print("length: " + str(len(dataset)))
for i in range(20):
    gan.fit(images)
    ganerateImages(gan.classGenerator,(5,5),gan.path + "\\progress" + str(i))
    gan.saveModel()