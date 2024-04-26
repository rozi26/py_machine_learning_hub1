import sys
sys.path.append("F:\\programing\\python\\machine_learning_hub")


from Data.Files_meneger import folderExist, createDir,getAvailableFolder
from Graphics.ShowImages import ganerateImages
from Models.GAN1_MC.Discriminator import Discriminator
from Models.GAN1_MC.Generator import Generator

from keras import optimizers,losses
import keras
import tensorflow as tf

GAN_SAVE_FOLDER = "F:\\programing\\python\\machine_learning_hub\\_not code_\\GEN_MODELS\\"

class GAN2(keras.Model):
    def __init__(self, discriminator, generator, generator_input, image_size, classes = 2, name="nameless"):
        super(GAN2, self).__init__()
        
        if(type(discriminator) == Discriminator): self.discriminator = discriminator.model
        else: self.discriminator = discriminator
        
        if(type(generator) == Generator): self.generator = generator.model
        else: self.generator = generator
        
        self.latent_dim = generator_input #the generator input size
        self.image_size = image_size # the size of the image
        self.optains = classes # how many optains there are to calssified the data
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")
        
        self.path =  getAvailableFolder(GAN_SAVE_FOLDER + name)
        self.training_uploads = 0
        self.classGenerator = Generator(self.generator,self.latent_dim + classes)

    def compile1(self):
        self.compile(
            d_optimizer=optimizers.Adam(learning_rate=0.0003),
            g_optimizer=optimizers.Adam(learning_rate=0.0003),
            loss_fn= losses.BinaryCrossentropy(from_logits=True)
        )

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN2, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        # Unpack the data.
        real_images, one_hot_labels = data

        # Add dummy dimensions to the labels so that they can be concatenated with
        # the images. This is for the discriminator.
        image_one_hot_labels = one_hot_labels[:, :, None, None]
        image_one_hot_labels = tf.repeat(
            image_one_hot_labels, repeats=[self.image_size * self.image_size]
        )
        image_one_hot_labels = tf.reshape(
            image_one_hot_labels, (-1, self.image_size, self.image_size, self.optains)
        )

        # Sample random points in the latent space and concatenate the labels.
        # This is for the generator.
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_vector_labels = tf.concat(
            [random_latent_vectors, one_hot_labels], axis=1
        )

        # Decode the noise (guided by labels) to fake images.
        generated_images = self.generator(random_vector_labels)

        # Combine them with real images. Note that we are concatenating the labels
        # with these images here.
        fake_image_and_labels = tf.concat([generated_images, image_one_hot_labels], -1)
        real_image_and_labels = tf.concat([real_images, image_one_hot_labels], -1)
        combined_images = tf.concat(
            [fake_image_and_labels, real_image_and_labels], axis=0
        )

        # Assemble labels discriminating real from fake images.
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )

        # Train the discriminator.
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space.
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_vector_labels = tf.concat(
            [random_latent_vectors, one_hot_labels], axis=1
        )

        # Assemble labels that say "all real images".
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_vector_labels)
            fake_image_and_labels = tf.concat([fake_images, image_one_hot_labels], -1)
            predictions = self.discriminator(fake_image_and_labels)
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        #save image
        #self.training_uploads += 1
        #ganerateImages(self.classGenerator,(5,5),self.path + "\\progress" + str(self.training_uploads))

        # Monitor loss.
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)
        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
        }
    
    def saveModel(self):
        folder = self.path + "\\models_save"
        if(not folderExist(folder)): createDir(folder)
        self.generator.save(folder + "\\generator")
        self.discriminator.save(folder + "\\discriminator")