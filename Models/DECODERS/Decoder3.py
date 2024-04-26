import numpy as np

from keras import Sequential
from keras.layers import Embedding, Flatten
from Data.ImageFolders import ReadImageDiractry2

DECODER_SAVE_FOLDER = "F:\\programing\\python\\machine_learning_hub\\_not code_\\DECORDERS\\"


def train(decoder:Sequential, images:ReadImageDiractry2):
    code_length = decoder.input_shape[0]

    embedder_model = Sequential()
    embedder_model.add(Embedding(images.bactch_length,code_length,input_length=1))
    embedder_model.add(Flatten())
    
    model = Sequential()
    model.add(embedder_model)
    model.add(decoder)
    

    for part in range(images.buffers):
        x_train = np.expand_dims(np.arange(code_length),axis=1)
        y_train = images.loadBatch(part)
        model.fit(x_train,y_train,batch_size=8)

