import sys
sys.path.append("F:\\programing\\python\\machine_learning_hub")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

from keras.models import Sequential, load_model
from Models.DECODERS.Models import get_decoder_218X178, get_encoder_218X178
from Graphics.ShowImages import showImage, showImages
from Data.ImageFolders import ReadImageDiractry2
import numpy as np
import keras
callback = keras.callbacks.TensorBoard(
    log_dir="F:\\storge\\board_dir", histogram_freq=1,
)

def add1(arr):
    return np.reshape(arr,((1,) + arr.shape))

dec = get_decoder_218X178(10,8000)
enc = get_encoder_218X178(8000)

#dec = load_model("F:\\programing\\python\\machine_learning_hub\\_not code_\\DECORDERS\\enc-dec_8\\saves\\decoder")
#enc = load_model("F:\\programing\\python\\machine_learning_hub\\_not code_\\DECORDERS\\enc-dec_8\\saves\\encoder")

dir = ReadImageDiractry2("F:\\storge\\face_dataset\\img_algin_c1",4000)
img1 = add1(dir.loadBatch(0)[0])
img2 = add1(dir.loadBatch(0)[1])

arr1 = enc.predict(img1)
arr2 = enc.predict(img2)

arr1 = add1(arr1)
arr2 = add1(arr2)
arr1 = add1(arr1)
arr2 = add1(arr2)

print(arr1)
print(arr2)

res1 = dec.predict(arr1)
res2 = dec.predict(arr2)

showImage(res1[0])
showImage(res2[0])

while(False):
    inp = np.random.randn(80)
    inp = np.reshape(inp,((1,) + inp.shape))
    img = dec.predict(inp,callbacks=callback)
    showImage(img[0])

"""comb = Sequential()
comb.add(enc)
comb.add(dec)

def add1(arr):
    return np.reshape(arr,((1,) + arr.shape))

dir = ReadImageDiractry2("F:\\storge\\face_dataset\\img_algin_c1",4000)


inp = dir.loadBatch(0)[0]
inp2 = add1(inp)
inp = add1(inp)


print(np.array_equal(inp,inp2))
p1 = comb.predict(inp)
p2 = comb.predict(inp2)
print(np.array_equal(p1,p2))


print(comb.summary())"""