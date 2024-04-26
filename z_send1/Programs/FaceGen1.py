import sys
sys.path.append("F:\\programing\\python\\machine_learning_hub")

import tkinter as tk
import numpy as np
import math
import cv2
import keyboard

from keras.models import load_model
from PIL import ImageTk, Image
from PIL.Image import fromarray
from cv2 import resize
from Data.ImageFolders import ReadImageDiractry2
from CutFace import cutFace, backgroundResize, removeBackground

#contolls
PAGE_WIDTH = 1800
PAGE_HEIGHT = 1000

MODELS_PATH = "F:\\programing\\python\\machine_learning_hub\\_not code_\\DECORDERS\\enc-dec_20\\saves_35\\"

SLIDERS_IN_LINE = 15
SLIDERS_MARGIN_X = 10
SLIDERS_MARGIN_Y = 20
SLIDERS_RANGE = 200

IMAGE_MARGIN = 25
IMAGE_SHOW_MULTYPLY = 2

BUTTON_HEIGHT = 40
BUTTON_MARGIN = 10

#calculate from contorls
ENCODER = load_model(MODELS_PATH + "encoder") #get the encoder
DECODER = load_model(MODELS_PATH + "decoder") #get the decoder

#calucalte image propaties
IMAGE_SIZE = (DECODER.output_shape[-3],DECODER.output_shape[-2]) #get the size of the image the machine learning work with
IMAGE_SHOW_SIZE = (IMAGE_SIZE[0] * IMAGE_SHOW_MULTYPLY,IMAGE_SIZE[1] * IMAGE_SHOW_MULTYPLY)

#calcualte sliders propaties
SLIDERS = ENCODER.output_shape[-1] #get the number of sliders
SLIDERS_WIDTH = round((PAGE_WIDTH - (IMAGE_SHOW_SIZE[1] + IMAGE_MARGIN * 2 + SLIDERS_MARGIN_X * (SLIDERS_IN_LINE + 2))) / SLIDERS_IN_LINE)
SLIDERS_HEIGHT = round((PAGE_HEIGHT - SLIDERS_MARGIN_Y * (math.ceil(SLIDERS / SLIDERS_IN_LINE) + 2)) / math.ceil(SLIDERS / SLIDERS_IN_LINE))

def getSliderX(id):
    return SLIDERS_MARGIN_X + (id % SLIDERS_IN_LINE) * (SLIDERS_MARGIN_X + SLIDERS_WIDTH)
def getSliderY(id):
    return SLIDERS_MARGIN_Y + (id // SLIDERS_IN_LINE) * (SLIDERS_MARGIN_Y + SLIDERS_HEIGHT)

print("slider width: " + str(SLIDERS_WIDTH) + "\nslider height: " + str(SLIDERS_HEIGHT))

#definae varibales
slider_vals = []
global show_image
show_image = None

#build the window
root = tk.Tk()
root.geometry(str(PAGE_WIDTH) + "x" + str(PAGE_HEIGHT))

#build the image
IMAGE_CANVES = tk.Canvas(root,width=IMAGE_SHOW_SIZE[1],height=IMAGE_SHOW_SIZE[0])
IMAGE_CANVES.place(x=PAGE_WIDTH - IMAGE_SHOW_SIZE[1] - IMAGE_MARGIN,y=IMAGE_MARGIN)

#define methods
def drawSeedsInImage(seeds, update_sliders = False):
    arr = DECODER.predict(np.reshape(seeds,(1,1,1,SLIDERS)),verbose=0)[0]
    arr = (arr * 255).astype(np.uint8)
    arr = np.reshape(arr,(IMAGE_SIZE + (3,)))
    #arr = np.ones((IMAGE_SHOW_SIZE + (3,)),dtype=np.uint8) * 120
    arr = resize(arr,(IMAGE_SHOW_SIZE[1],IMAGE_SHOW_SIZE[0]))
    global show_image
    show_image = ImageTk.PhotoImage(image=fromarray(arr))
    IMAGE_CANVES.create_image((0,0),anchor=tk.NW,image=show_image)

    if(update_sliders):
        for i in range(SLIDERS):
            slider_vals[i].set(seeds[i])
    #IMAGE_CANVES.create_rectangle(0,0,IMAGE_SHOW_SIZE,fill="red")

def slider_update(event):
    arr = np.zeros(SLIDERS)
    for i in range(len(slider_vals)):
        arr[i] = slider_vals[i].get()
    drawSeedsInImage(arr)

def encode_image(image):
    image = np.reshape(image,(1,) + image.shape)
    enc = ENCODER.predict(image)
    print(enc)
    for i in range(SLIDERS):
        slider_vals[i].set(enc[0][0][0][i])
    drawSeedsInImage(enc)

def take_a_picture():
    cap = cv2.VideoCapture(0)
    """CAMERA_WIDTH = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    CAMERA_HEIGHT = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print("camera resulution (" + str(CAMERA_WIDTH) + "," + str(CAMERA_HEIGHT) +")")
    CAMERA_RATIO = CAMERA_HEIGHT / CAMERA_WIDTH
    IMAGE_RATIO = IMAGE_SIZE[0] / IMAGE_SIZE[1]
    WINDOW_HEIGHT = CAMERA_HEIGHT if (IMAGE_RATIO >= CAMERA_RATIO) else math.ceil(IMAGE_SIZE[0] * CAMERA_WIDTH) / IMAGE_SIZE[1]
    WINDOW_WIDTH = CAMERA_WIDTH if (IMAGE_RATIO <= CAMERA_RATIO) else math.ceil(IMAGE_SIZE[1] * CAMERA_HEIGHT) / IMAGE_SIZE[0]

    print("page size (" + str(WINDOW_WIDTH) + "," + str(WINDOW_HEIGHT) +") ratio: " + str(WINDOW_HEIGHT / WINDOW_WIDTH) + " original ratio: " + str(IMAGE_RATIO))
    """
    face = None
    while(True):
        ret, frame = cap.read()
        face,found = cutFace(frame)
        #face = cv2.resize(face,(IMAGE_SIZE[1],IMAGE_SIZE[0]))
        face = backgroundResize(face,IMAGE_SIZE)
        cv2.imshow("img",face)
        cv2.waitKey(1)
        if(keyboard.is_pressed('q') or (keyboard.is_pressed('d') and found)):
            break
    cap.release()
    cv2.destroyAllWindows()
    face = face[:,:,::-1]
    print("face shape " + str(face.shape))
    cv2.imshow("your face",face)
    encode_image(face / 255.0)

def fill_with_random():
    seeds = np.random.randn(SLIDERS) * 40
    drawSeedsInImage(seeds,update_sliders=True)

def fill_with_file(path, cut = True):
    arr = np.array(Image.open(path))[:,:,::-1]
    arr = np.array(arr,dtype=np.uint8)
    face, found = cutFace(arr)
    face = backgroundResize(face,IMAGE_SIZE)
    face = face[:,:,::-1]
    #face = resize(face,(IMAGE_SIZE[1],IMAGE_SIZE[0]))
    cv2.imshow("test",face)
    encode_image(face / 255.0)

#build the sliders
for i in range(SLIDERS):
    var = tk.DoubleVar()
    slider = tk.Scale(root,variable=var,from_=-SLIDERS_RANGE,to=SLIDERS_RANGE,length=SLIDERS_HEIGHT, width=SLIDERS_WIDTH,command=slider_update,resolution=0.00001)
    slider.place(x=getSliderX(i),y=getSliderY(i))
    slider_vals.append(var)

#build camera build
camera_batton = tk.Button(root,text="edit yourself",command=lambda:take_a_picture())
camera_batton.place(x=PAGE_WIDTH - IMAGE_SHOW_SIZE[1] - IMAGE_MARGIN, y = IMAGE_SHOW_SIZE[0] + IMAGE_MARGIN * 2,width=IMAGE_SHOW_SIZE[1],height=BUTTON_HEIGHT)

random_batton = tk.Button(root,text="random face",command=lambda:fill_with_random())
random_batton.place(x=PAGE_WIDTH - IMAGE_SHOW_SIZE[1] - IMAGE_MARGIN, y = IMAGE_SHOW_SIZE[0] + IMAGE_MARGIN * 2 + BUTTON_HEIGHT + BUTTON_MARGIN,width=IMAGE_SHOW_SIZE[1],height=BUTTON_HEIGHT)

der = ReadImageDiractry2("F:\\storge\\face_dataset\\img_algin_c1",2000)
img = der.loadImage(name="000814.jpg")
cv2.imshow("im1",img)
#encode_image(img)

fill_with_file("F:\\downloads\\emma-watson-chanel-harry-potter-and-the-order-of-the-phoenix-premiere-copy.jpg",True)
#fill_with_file("F:\\storge\\face_dataset\\img_algin_c1\\000019.jpg",False)

print("image show size: " + str(IMAGE_SHOW_SIZE))
print("image size is: " +str(IMAGE_SIZE))
tk.mainloop()

