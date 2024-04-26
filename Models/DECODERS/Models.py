
#keras imports
from keras.models import Sequential
from keras.layers import Conv2DTranspose, Reshape, Embedding, Activation, Flatten, Input, Conv2D, Dense
from keras.optimizers import Adam

#tensorflow settings

#paramets
LEARNING_RATE = 0.0005

def get_decoder_218X178(num_sampels, parameters):
    model = Sequential()

    """model.add(Input((1,parameters)))
    model.add(Flatten())
    #print (model.output_shape)

    model.add(Reshape((1,1,80), name='decoder'))
    #model.add(Reshape((1,1,80)))
    print (model.output_shape)"""
    model.add(Input((1,1,parameters),name="decoder_start"))

    model.add(Conv2DTranspose(128, (17, 12)))           #(4, 1)
    model.add(Activation("relu"))
    print (model.output_shape)

    model.add(Conv2DTranspose(128, 4))                #(7, 4)
    model.add(Activation("relu"))
    print (model.output_shape)
    
    model.add(Conv2DTranspose(128, 4))                #(10, 7)
    model.add(Activation("relu"))
    print( model.output_shape)

    model.add(Conv2DTranspose(128, 4, strides=2))     #(22, 16)
    model.add(Activation("relu"))
    print (model.output_shape)

    model.add(Conv2DTranspose(128, 4))                #(10, 7)
    model.add(Activation("relu"))
    print( model.output_shape)
    
    
    model.add(Conv2DTranspose(128, 4, strides=2))     #(46, 34)
    model.add(Activation("relu"))
    print (model.output_shape)

    model.add(Conv2DTranspose(3,12, strides=2))     #(46, 34)
    model.add(Activation("sigmoid"))
    


    print (model.output_shape[1:])
    #assert(model.output_shape[1:] == (3, 192, 144))

    #model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse')
    return model

def get_encoder_218X178(parameters):
    model = Sequential()

    model.add(Input((218,178,3)))
    #model.add(Flatten(name='pre_encoder'))
    model.add(Conv2D(128,4,1))

    model.add(Conv2D(128,4,2))

    model.add(Conv2D(128,4,2))
    
    model.add(Conv2D(128,4,2))

    model.add(Conv2D(64,4,2))

    model.add(Flatten())
    model.add(Dense(parameters))
    model.add(Reshape((1,1,parameters)))
    #model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse')
    return model

def compile1(model):
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse')