import sys
sys.path.append("F:\\programing\\python\\machine_learning_hub")


import numpy as np
from Models.DECODERS.Decoder2 import Decorder, DECODER_SAVE_FOLDER
from Models.DECODERS.Models import get_decoder_218X178, get_encoder_218X178
from Data.ImageFolders import ReadImageDiractry2
from Graphics.ShowImages import showImages, saveImages, showImage
from keras.models import load_model

import keras
callback = keras.callbacks.TensorBoard(
    log_dir="F:\\storge\\board_dir", histogram_freq=1,
)

#LOAD_FROM = "F:\\programing\\python\\machine_learning_hub\\_not code_\\DECORDERS\\enc-dec_31\\saves_50\\"
LOAD_FROM = None
enc = None
dec = None
if(LOAD_FROM == None):
    enc = get_encoder_218X178(80)
    dec = get_decoder_218X178(100,80)
else:
    enc = load_model(LOAD_FROM + "encoder")
    dec = load_model(LOAD_FROM + "decoder")

d2 = Decorder(enc,dec,"enc-dec")

"""d2.model.build((None,218,178,3))
print(d2.model.summary())
print(d2.model.input_shape)
print(d2.encoder.output_shape)
print(d2.decoder.input_shape)

from keras.utils import plot_model
plot_model(d2.model,"F:\\programing\\python\\machine_learning_hub\\_not code_\\DECORDERS\\model.png",show_shapes=True)
plot_model(d2.encoder,"F:\\programing\\python\\machine_learning_hub\\_not code_\\DECORDERS\\encoder.png",show_shapes=True)
plot_model(d2.decoder,"F:\\programing\\python\\machine_learning_hub\\_not code_\\DECORDERS\\decoder.png",show_shapes=True)"""

buffers = 500
dir = ReadImageDiractry2("F:\\storge\\face_dataset\\img_algin_c1",buffers)

val_data = None
#set val data
#val_data = dir.loadBatch(buffers - 1)
#val_data = (val_data,val_data


test_im = dir.loadBatch(0)[0:10]
for i in range(buffers):
    
    batch = dir.loadBatch(i)
    d2.train(batch,batch_size=48,callback=callback)

    if(i % 10 == 0 ):
        d2.save(i % 10 == 0)
    save = batch[:25]
    save[0:10] = test_im
    saveImages(d2.generateFromImages(save,(5,5)),d2.path + "\\images\\i" + str(i))

d2.save()

def in_out_modfe(model):
    print("input shape: " + str(model.input_shape) + "\noutput shape: " + str(model.output_shape))


decoder = load_model("F:\\programing\\python\\machine_learning_hub\\_not code_\\DECORDERS\\enc-dec_8\\saves\\decoder")
#input = np.zeros((1,80))
in_out_modfe(decoder)

while(True):
    input = np.random.randn((80))
    input = np.reshape(input,(1,80))
    img = decoder.predict(input)
    showImage(img[0])

dec = get_decoder_218X178(10,80)
