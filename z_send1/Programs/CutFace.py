import cv2
import numpy as np
import mediapipe as mp

from cv2.data import haarcascades

BACKGROUND_IMAGE = "F:\\downloads\\download (1).png"

def getFaces(img, exp = 0.4, MIN_SIZE = 100):
    original = img.copy()
    def markDerc(dc, v1=1.2,v2=2, color=(255,0,0), expentain=0, draw=True):
        mark = dc.detectMultiScale(img,v1,v2)

        if(expentain != 0):
            for i in range(len(mark)):
                (x,y,w,h) = mark[i]
                e_x = w * expentain
                e_y = h * expentain
                mark[i] = (max(round(x-e_x),0),max(round(y-e_y),0),min(w + e_x * 2,len(img[0])),min(h + e_y * 2,len(img)))
        if(draw):
            for (x, y, w, h) in mark:
                cv2.rectangle(img,(x,y),(min(x+w,len(img[0])),min(y+h,len(img))),color,2)
               # cv2.rectangle(img, (max(round(x - e_x),0), max(round(y-e_y),0)), (min(round(x + w + e_x),len(img[0])),min(round(y+h+e_y),len(img))), color, 2)
        return mark
    dc1 = cv2.CascadeClassifier(haarcascades +  "haarcascade_frontalface_default.xml")
    dc2 = cv2.CascadeClassifier(haarcascades + "haarcascade_frontalface_alt2.xml")

    markDerc(dc1,1.2,2)

    
    marks = markDerc(dc2,1.1,2,(0,0,255),exp)

    faces = []
    for (x,y,w,h) in marks:
        if(w >= MIN_SIZE and h >= MIN_SIZE):
            cut = original[y:y+h,x:x+w]
            faces.append(cut)
    return faces


def removeBackground(img):
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
    height , width, channel = img.shape
    RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = selfie_segmentation.process(RGB)

    bg_image = cv2.imread(BACKGROUND_IMAGE)
    condition = np.stack(
    (results.segmentation_mask,) * 3, axis=-1) > 0.5
    # resize the background image to the same size of the original frame
    bg_image = cv2.resize(bg_image, (width, height))
    # combine frame and background image using the condition
    output_image = np.where(condition, img, bg_image)

    return output_image

def cutFace(image):
    image = removeBackground(image)
    faces = getFaces(image)
    if(len(faces) == 0):
        return cv2.imread(BACKGROUND_IMAGE), False
    face = faces[0]
    return face, True

def backgroundResize(image, size):
    image_height = len(image)
    image_width = len(image[0])
    
    if(image_height > size[0] or image_width > size[1]):
       down = min(size[0] / image_height,size[1] / image_width)
       image_height = round(image_height * down)
       image_width = round(image_width * down)
       image = cv2.resize(image,(image_width,image_height))
    bg_image = np.array(cv2.imread(BACKGROUND_IMAGE))[0:size[0],0:size[1]]
    print("\r reshape for brg: " + str(bg_image.shape) + "  (size " + str(size) + " )   ",end="")
    top_margin = len(bg_image) - image_height
    left_margin = (len(bg_image[0]) - image_width) // 2

    bg_image[top_margin:top_margin + image_height,left_margin:left_margin + image_width] = image
    return bg_image

