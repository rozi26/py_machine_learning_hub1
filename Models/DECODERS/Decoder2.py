import sys
sys.path.append("F:\\programing\\python\\machine_learning_hub")

from Data.Files_meneger import getAvailableFolder, folderExist, createDir
from Graphics.ShowImages import saveImage
from Models.DECODERS.Models import compile1
from keras.models import Sequential

DECODER_SAVE_FOLDER = "F:\\programing\\python\\machine_learning_hub\\_not code_\\DECORDERS\\"
class Decorder():

    def __init__(self, encoder, decoder, name="namless"):
        
        self.encoder = encoder
        self.decoder = decoder

        self.model = Sequential()
        self.model.add(encoder)
        self.model.add(decoder)
        compile1(self.model)

        self.name = name
        self.path = None

        self.images = 0
    
    def save(self, newFile=False):
        if(self.path == None):
            self.path = getAvailableFolder(DECODER_SAVE_FOLDER + self.name)
            createDir(self.path + "\\images")
        current_path = ""
        if(newFile):
            current_path = getAvailableFolder(self.path + "\\saves")
        else:
            current_path = self.path + "\\saves"
        
        if(not folderExist(current_path)):
            createDir(current_path)

        self.model.save(current_path + "\\model")
        self.encoder.save(current_path + "\\encoder")
        self.decoder.save(current_path + "\\decoder")
        
    
    def train(self, images, batch_size=8,epochs=1, callback=None, val_data=None):
        losses = []
        for epoch in range(epochs):

            history = self.model.fit(images,images,batch_size=batch_size,epochs=6,verbose=1,callbacks=[callback])
            
            loss = history.history['loss'][-1]
            losses.append(loss)


    def generateFromImages(self, images,shape):
        data = images[:(shape[0] * shape[1])]
        encodes = self.encoder.predict(data)
        results = self.decoder.predict(encodes)

        #results = self.model.predict(images[:(shape[0] * shape[1])])
        results = results.reshape((shape + self.model.output_shape[1:]))
        return results

    def generateFromSeed(self, seed):
        return self.decoder.predict(seed)