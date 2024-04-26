import sys
sys.path.append("F:\\programing\\python\\machine_learning_hub")

from Data.ImageFolders import ReadImageDiractry
from Graphics.ShowImages import showImage
from Graphics.ProgressBar import ProgressBar
from Models.GAN1_MC.Generator import modelG_keras_64x64_1,modelG_keras_178X218_1, Generator
from Models.GAN1_MC.Discriminator import model_keras_64x64_1,model_keras_178X218_1, Discriminator
from Models.GAN1_MC.GANS.GAN1 import GAN
from keras.models import load_model
import numpy as np

#PATH1 = "F:\\storge\\face_dataset\\00000\\"
PATH1 = "F:\\storge\\face_dataset\\img_algin_c1\\"

BUFFERS = 1000
GENERATOR_INPUT = 64

data = ReadImageDiractry(PATH1,buffers=BUFFERS,image_size=(218,178),convetion=1)

model_d = model_keras_178X218_1()
model_g = modelG_keras_178X218_1(GENERATOR_INPUT)
#model_g = load_model("F:\\programing\\python\\machine_learning_hub\\_not code_\\GEN_MODELS\\face generator_2\\models_save\\generator")
gan = GAN(model_g,model_d,generator_input_shape=GENERATOR_INPUT,discriminator_input_shape=(218,178,3),name="face generator")

progress = ProgressBar(BUFFERS)
for i in range(BUFFERS):
    progress.progress()
    load = data.loadBatch(i)
    gan.train(load)
gan.save()
print("done")
