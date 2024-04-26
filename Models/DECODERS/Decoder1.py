import sys
sys.path.append("F:\\programing\\python\\machine_learning_hub")


from Data.Files_meneger import getAvailableFolder, folderExist, createDir
from Graphics.ShowImages import saveImage


DECODER_SAVE_FOLDER = "F:\\programing\\python\\machine_learning_hub\\_not code_\\DECORDERS\\"
class Decoder():

    def __init__(self, model, name="nameless"):
        self.model = model

        self.name = name
        self.path = None
        self.images = 0
    
    def save(self, newFile):
        if(self.path == None):
            self.path = getAvailableFolder(DECODER_SAVE_FOLDER + self.name)
        if(newFile):
            path = getAvailableFolder(self.path + "\\model")
            self.model.save(path)
            return
        self.model.save(self.path + "\\model")
    
    def train(self, x_train, y_train, batch_size=8,epochs=1, save_frequancy=20):
        train_loss = []
        for iters in range(epochs):
            history = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=1,verbose=1)
            
            loss = history.history['loss'][-1]
            train_loss.append(loss)

            if(iters % save_frequancy == 0):
                self.save(newFile=True)

                if(not folderExist(self.path + "\\images")):
                    createDir(self.path + "\\images")
                y_faces = self.model.predict(x_train[:25], batch_size=batch_size)
                for i in range(y_faces.shape[0]):
                    saveImage(y_faces[i],self.path + "\\images\\i" + str(self.images))
                    self.images += 1
