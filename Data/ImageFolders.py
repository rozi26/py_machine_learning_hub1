import numpy as np

from os.path import exists
from os import listdir
from pickletools import uint8
from PIL import Image
from cv2 import resize

class ReadImageDiractry():
    
    def __init__(self, path,buffers=1, image_size=None,imageSizeEqual=True, convetion=0):
        if(not exists(path)):
            print("the folder in path: '" + str(path) + "' doesn't exist")
            return
        self.path = path #the folder path
        self.dir = listdir(path);#the files names
        self.length = len(self.dir)#the number of files
        self.buffers = buffers#of many buffers there are
        self.banch = self.length // buffers #banch length
        self.sameSize = imageSizeEqual#if all the images in the folder are in the same size
        self.convert = convetion #0-[0:255],1-[0:1],2-[-1:1]
        
        shape = np.array(Image.open(self.path + self.dir[0])).shape
        self.size = image_size if (image_size != None)else (shape[0],shape[1])   #the resize image size
        #print("shape: " + str(shape) + "\nsize: " + str(self.size) + " (" + str(shape[1] % self.size[1]) + ")")
        self.colors = shape[2] if len(shape) > 2 else 1
        self.cutSkip = None
        if(imageSizeEqual and image_size != None):
            shape = np.array(Image.open(self.path + self.dir[0])).shape
            if(shape[0] % self.size[0] == 0 and shape[1] % self.size[1] == 0):
                self.cutSkip = (shape[0] / self.size[0],shape[1] / self.size[1])
  
    def loadBatch(self, batch_id):
        start = batch_id * self.banch
        end = start + self.banch if (not batch_id == self.buffers - 1) else self.length
        shape = (end - start,self.size[0],self.size[1],self.colors)# if (self.colors != 1) else (end - start,self.size[0],self.size[1])
        imgs = np.zeros(shape,dtype=np.uint8)
        
        loc = 0
        #print(str(start) + " - " + str(end))
        if(self.cutSkip == None or not self.sameSize):
            for i in range(start,end):
                imgs[loc] = resize(np.array(Image.open(self.path + self.dir[i])),self.size)
                loc += 1
        elif(self.cutSkip[0] == 1 and self.cutSkip[1] == 1):
            for i in range(start,end):
                arr = np.array(Image.open(self.path + self.dir[i]))
                imgs[loc] = arr.reshape(arr.shape)
                loc += 1
        else:
            for i in range(start,end):
                imgs[loc] = np.array(Image.open(self.path + self.dir[i]))[::self.cutSkip[0],::self.cutSkip[1]]
                loc += 1
        
        if(self.convert == 1): return imgs.astype('float32') / 255.0
        if(self.convert == 2): return (imgs.astype('float32') - 127.5) / 127.5
        return imgs
    
class ReadImageDiractry2():
    def __init__(self, path, buffers=1,output_shape=None, diffrent_size_images = False):
        if(not exists(path)):
            print("can't read from image diractry because there is no fodler at [" + str(path) + "]")
            return
        if(path[-1] != "\\"): path += "\\"
        self.path = path
        self.buffers = buffers

        self.dir = listdir(path)
        self.length = len(self.dir)
        self.bactch_length = self.length // buffers
        self.first_bactch_length = self.bactch_length + self.length % self.bactch_length

        #shape meneging
        self.input_shape = np.array(Image.open(self.path + self.dir[0])).shape
        self.output_shape = self.input_shape if output_shape==None else (output_shape if (len(output_shape) == 3) else output_shape + (self.input_shape[2],))
        self.reshape_method = 0 if (not diffrent_size_images and self.input_shape == self.output_shape) else 1

    def loading_reshape(self,img):
        if(self.reshape_method == 0):return img
        return resize(img,(self.output_shape[1],self.output_shape[0]))    
    def loadBatch(self, batch_id):
        start = self.first_bactch_length + self.bactch_length * (batch_id - 1) if batch_id != 0 else 0
        end = start + (self.first_bactch_length if batch_id == 0 else self.bactch_length)
        
        print("load from " + str(start) + " to " + str(end))
        
        arr = np.zeros(((end - start,) + self.output_shape))

        for i in range(start,end):
            arr[i - start] = self.loading_reshape(np.array(Image.open(self.path + self.dir[i])))
        arr = arr / 255.0
        return arr

    def loadImage(self, id=None,name=None):
        if(id != None):
            return self.loading_reshape(np.array(Image.open(self.path + self.dir[id]))) / 255.0
        if(name != None):
            return self.loading_reshape(np.array(Image.open(self.path + name))) / 255.0
        print("read image from diractry error. when you use loadImage method you must set id or name")
        return None