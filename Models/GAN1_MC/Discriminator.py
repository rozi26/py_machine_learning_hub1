from keras import layers
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
import keras
class Discriminator():
    def __init__(self, model, input_size):
        self.model = model
        self.input = input_size

def model_keras_64x64_1():
    discriminator = keras.Sequential(
    [
        keras.Input(shape=(64, 64, 3)),
        layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ],
    name="discriminator",
    )
    discriminator.compile(optimizer=Adam(learning_rate=0.0001),loss=BinaryCrossentropy())
    return discriminator


def model_keras_28x28_1():
    discriminator = keras.Sequential(
    [
        keras.layers.InputLayer((28,28,3)),
        layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, (3, 3), strides=(4, 4), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.GlobalMaxPooling2D(),
        layers.Dense(1),
    ],
    name="discriminator",)
    discriminator.compile(optimizer=Adam(lr=0.0002),loss=BinaryCrossentropy())
    return discriminator

def model_keras_178X218_1():
    discriminator = keras.Sequential(
    [
        keras.layers.InputLayer((218,178,3)),
        layers.Conv2D(32, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(48, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Flatten(),
        layers.Dense(1, activation="sigmoid"),
    ],
    name="discriminator",)
    discriminator.compile(optimizer=Adam(lr=0.0002),loss=BinaryCrossentropy())
    return discriminator


