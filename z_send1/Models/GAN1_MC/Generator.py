from keras import layers
import keras
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy


import numpy as np

class Generator():
    
    def __init__(self, model, input_size, output_size=None):
        self.model = model
        self.input = input_size
        print("input: " + str(self.input))
        
    def generate(self, seed=None):
        if(seed == None):
            seed = self.create_random_input()
        return self.model.predict(seed,verbose=0)
    
    def generate_list(self, amount):
        return self.model.predict(self.create_random_input(amount),verbose=0)
    
    def create_random_input(self, amount=None):
        if(amount == None): return np.random.randn(self.input).reshape(1,self.input)
        return np.random.randn(amount,self.input)



def modelG_keras_64x64_1(input_length):
    generator = keras.Sequential(
    [
        keras.Input(shape=(input_length,)),
        layers.Dense(8 * 8 * 128),
        layers.Reshape((8, 8, 128)),
        layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(3, kernel_size=5, padding="same", activation="sigmoid"),
    ],
    name="generator",)
    generator.compile(optimizer=Adam(learning_rate=0.0001),loss=BinaryCrossentropy())
    return generator

def modelG_keras_28x28_1(input_length):
    generator = keras.Sequential(
    [
        keras.layers.InputLayer((input_length,)),
        # We want to generate 128 + num_classes coefficients to reshape into a
        # 7x7x(128 + num_classes) map.
        layers.Dense(7 * 7 * input_length),
        layers.LeakyReLU(alpha=0.2),
        layers.Reshape((7, 7, input_length)),
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid"),
    ],
    name="generator",)
    #generator.compile(optimizer=Adam(learning_rate=0.0001),loss=BinaryCrossentropy())
    return generator

def modelG_keras_178X218_1(input_length):
    generator = keras.Sequential(
    [
        keras.Input(shape=(input_length,)),
        layers.Dense(4 * 4 * 128),
        layers.Reshape((4, 4, 128)),
        layers.Conv2DTranspose(128, kernel_size=4, strides=3, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(128, kernel_size=4, strides=3, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
        layers.Conv2D(3, kernel_size=5, padding="same"),
        layers.Resizing(218,178)
    ],
    name="generator",)
    generator.compile(optimizer=Adam(learning_rate=0.0001),loss=BinaryCrossentropy())
    return generator
