import numpy as np
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dropout, Activation
from keras.models import Model, Sequential
from keras import backend as K
import keras
from keras.datasets import mnist

from MyOptimizers import AdamPlus,AMSGradPlus, AdaMaxPlus, Adam,AMSGrad,AdaMax


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1)) 
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

nepochs = 1 # num of epochs

# building network
def create_model():
    # codes from blog.keras.io/building-autoencoders-in-keras.html
    model = Sequential() 
    input_img = Input(shape=(28, 28, 1))

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    return autoencoder


import os
directory = 'MNIST/'
if not os.path.exists(directory):
    os.makedirs(directory)

import time

N = 30 # number of trials
verbose_or_not = 1 # verbose or not

# Adam
loss = []
val = []
for i in range(N):
    trainer = Adam(lr = 0.00002)
    model = create_model()
    model.compile(optimizer=trainer, loss='mse')
    log = model.fit(x_train, x_train, epochs=nepochs, batch_size=128, 
                    shuffle=True, verbose=verbose_or_not, validation_data=(x_test,x_test))
    keras.backend.clear_session()
    loss.append(log.history['loss'])
    # val.append(log.history['val_loss'])
loss = np.array(loss)
# val = np.array(val)
np.savetxt(directory+'Adam_loss.txt',loss,fmt = '%.8f')
# np.savetxt(directory+'Adam_val.txt',val,fmt = '%.8f')

# Adam+
loss = []
val = []
for i in range(N):
    trainer = AdamPlus(lr=0.00002, mu=0.2)
    model = create_model()
    model.compile(optimizer=trainer, loss='mse')
    log = model.fit(x_train, x_train, epochs=nepochs, batch_size=128, 
                    shuffle=True, verbose=verbose_or_not, validation_data=(x_test,x_test))
    keras.backend.clear_session()
    loss.append(log.history['loss'])
    # val.append(log.history['val_loss'])
loss = np.array(loss)
# val = np.array(val)
np.savetxt(directory+'Adam+_loss.txt',loss,fmt = '%.8f')
# np.savetxt(directory+'Adam+_val.txt',val,fmt = '%.8f')

# AdaMax
loss = []
val = []
for i in range(N):
    trainer = AdaMax(lr = 0.00002)
    model = create_model()
    model.compile(optimizer=trainer, loss='mse')
    log = model.fit(x_train, x_train, epochs=nepochs, batch_size=128, 
                    shuffle=True, verbose=verbose_or_not, validation_data=(x_test,x_test))
    keras.backend.clear_session()
    loss.append(log.history['loss'])
    # val.append(log.history['val_loss'])
loss = np.array(loss)
# val = np.array(val)
np.savetxt(directory+'AdaMax_loss.txt',loss,fmt = '%.8f')
# np.savetxt(directory+'AdaMax+_val.txt',val,fmt = '%.8f')

# AdaMax+
loss = []
val = []
for i in range(N):
    trainer = AdaMaxPlus(lr=0.00002, mu=0.2)
    model = create_model()
    model.compile(optimizer=trainer, loss='mse')
    log = model.fit(x_train, x_train, epochs=nepochs, batch_size=128, 
                    shuffle=True, verbose=verbose_or_not, validation_data=(x_test,x_test))
    keras.backend.clear_session()
    loss.append(log.history['loss'])
    # val.append(log.history['val_loss'])
loss = np.array(loss)
# val = np.array(val)
np.savetxt(directory+'AdaMax+_loss.txt',loss,fmt = '%.8f')
# np.savetxt(directory+'AdaMax+_val.txt',val,fmt = '%.8f')

# AMSGrad
loss = []
val = []
for i in range(N):
    trainer = AMSGrad(lr = 0.00002)
    model = create_model()
    model.compile(optimizer=trainer, loss='mse')
    log = model.fit(x_train, x_train, epochs=nepochs, batch_size=128, 
                    shuffle=True, verbose=verbose_or_not, validation_data=(x_test,x_test))
    keras.backend.clear_session()
    loss.append(log.history['loss'])
    # val.append(log.history['val_loss'])
loss = np.array(loss)
# val = np.array(val)
np.savetxt(directory+'AMSGrad_loss.txt',loss,fmt = '%.8f')
# np.savetxt(directory+'AMSGrad_val.txt',val,fmt = '%.8f')

# AMSGrad+
loss = []
val = []
for i in range(N):
    trainer = AMSGradPlus(lr=0.00002, mu=0.2)
    model = create_model()
    model.compile(optimizer=trainer, loss='mse')
    log = model.fit(x_train, x_train, epochs=nepochs, batch_size=128, 
                    shuffle=True, verbose=verbose_or_not, validation_data=(x_test,x_test))
    keras.backend.clear_session()
    loss.append(log.history['loss'])
    # val.append(log.history['val_loss'])
loss = np.array(loss)
# val = np.array(val)
np.savetxt(directory+'AMSGrad+_loss.txt',loss,fmt = '%.8f')
# np.savetxt(directory+'AMSGrad+_val.txt',val,fmt = '%.8f')
