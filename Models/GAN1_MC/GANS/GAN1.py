import sys
sys.path.append("F:\\programing\\python\\machine_learning_hub")
from Graphics.ProgressBar import ProgressBar

from Models.GAN1_MC.Generator import Generator
from Models.GAN1_MC.Discriminator import Discriminator
import numpy as np

from keras import Sequential
from keras.optimizers import Adam

from Data.Files_meneger import folderExist, getAvailableFolder,createDir
from Graphics.ShowImages import*

GAN_SAVE_FOLDER = "F:\\programing\\python\\machine_learning_hub\\_not code_\\GEN_MODELS\\"

class GAN():
    def __init__(self, generator, discriminator, name = "nameless", folder=GAN_SAVE_FOLDER, generator_input_shape=None, discriminator_input_shape=None):
        self.model_g = generator if (type(generator) == Generator) else Generator(generator,generator_input_shape)
        self.model_d = discriminator if (type(discriminator) == Discriminator) else Discriminator(discriminator,discriminator_input_shape)
        self.model_d.model.trainable = False
        
        self.model = Sequential()
        self.model.add(self.model_g.model)
        self.model.add(self.model_d.model)
        
        #customable
        self.model.compile(optimizer=Adam(lr=0.0002,beta_1=0.5),loss="binary_crossentropy")
        
        #props
        self.path = getAvailableFolder(folder + name)
        self.evulotainSave = createDir(self.path + "\\progress")
        self.training_generation = 0
    
    def train(self, real_data, epochs=2,progressBar=None, saveProgress=True, save=True):
        batch_half = len(real_data)
        
        for i in range(epochs):
            x_real, y_real = real_data, np.ones((batch_half,1))
            x_fake, y_fake = self.model_g.generate_list(batch_half), np.zeros((batch_half,1))
            x_final, y_final = np.vstack((x_real,x_fake)), np.vstack((y_real,y_fake))
            
            self.model_d.model.trainable = True
            self.model_d.model.train_on_batch(x_final,y_final)
            self.model_d.model.trainable = False
            self.model.train_on_batch(self.model_g.create_random_input(batch_half * 2),np.ones((batch_half * 2,1)))
        
            if(progressBar != None):
                progressBar.progress()
            
        if(saveProgress):
            self.training_generation += 1
            ganerateImages(self.model_g,(5,5),self.evulotainSave + "\\run" + str(self.training_generation))
        
        if(save and self.training_generation % 10 == 0):
            self.save()
     
    def train_on_data(self, folder, batchSize = 256, epoches=5, showProgress=True):
        progressBar = ProgressBar(epoches * (folder.banch // batchSize) * folder.buffers) if showProgress else None
        helf_batch = batchSize // 2
        for i in range(folder.buffers):
            for e in range(epoches):
                data = folder.loadBatch(i)
                for g in range(len(data) // batchSize):
                    real = data[np.random.randint(0,len(data),helf_batch)]
                    self.train(real,epochs=1,saveProgress=False,save=False)
                    if(showProgress):  progressBar.progress()
                    
                self.training_generation += 1
                ganerateImages(self.model_g,(5,5),self.evulotainSave + "\\run" + str(self.training_generation))
                if(e % 5 == 0): self.save()
            self.save()
                
       
    def save(self):
        folder = self.path + "\\models_save"
        if(not folderExist(folder)): createDir(folder)
        self.model.save(folder + "\\GAN")
        self.model_g.model.save(folder + "\\generator")
        self.model_d.model.save(folder + "\\discriminator")
