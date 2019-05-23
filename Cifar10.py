import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras import backend as K
from keras.utils import np_utils

from keras.datasets import cifar10
from MyOptimizers import AdamPlus, AMSGradPlus, AdaMaxPlus, Adam, AMSGrad, AdaMax

batch_size = 128
num_classes = 10
nepochs = 20

(x_train,y_train),(x_test,y_test) = cifar10.load_data()

y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train  /= 255
x_test /= 255

# building network
def create_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=x_train.shape[1:]))
    model.add(Dropout(0.2))

    model.add(Conv2D(32,(3,3),padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
    model.add(Dropout(0.2))

    model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
    model.add(Dropout(0.2))

    model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(1024,activation='relu',kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    
    return model
    

import os
directory = 'Cifar10/'
if not os.path.exists(directory):
    os.makedirs(directory)

N = 1 # number of trials

# Adam
loss = []
val = []
for i in range(N):
    trainer = Adam(lr = 0.001)
    model = create_model()
    model.compile(optimizer=trainer, loss=keras.losses.categorical_crossentropy, 
                    metrics=['accuracy'])
    log = model.fit(x_train, y_train, epochs=nepochs, batch_size=128, 
                    shuffle=True, verbose=1, validation_data=(x_test,y_test))
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
    trainer = AdamPlus(lr=0.001, mu=0.7)
    model = create_model()
    model.compile(optimizer=trainer, loss=keras.losses.categorical_crossentropy, 
                    metrics=['accuracy'])
    log = model.fit(x_train, y_train, epochs=nepochs, batch_size=128, 
                    shuffle=True, verbose=1, validation_data=(x_test,y_test))
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
    trainer = AdaMax(lr = 0.001)
    model = create_model()
    model.compile(optimizer=trainer, loss=keras.losses.categorical_crossentropy, 
                    metrics=['accuracy'])
    log = model.fit(x_train, y_train, epochs=nepochs, batch_size=128, 
                    shuffle=True, verbose=1, validation_data=(x_test,y_test))
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
    trainer = AdaMaxPlus(lr=0.001, mu=0.7)
    model = create_model()
    model.compile(optimizer=trainer, loss=keras.losses.categorical_crossentropy, 
                    metrics=['accuracy'])
    log = model.fit(x_train, y_train, epochs=nepochs, batch_size=128, 
                    shuffle=True, verbose=1, validation_data=(x_test,y_test))
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
    trainer = AMSGrad(lr = 0.001)
    model = create_model()
    model.compile(optimizer=trainer, loss=keras.losses.categorical_crossentropy, 
                    metrics=['accuracy'])
    log = model.fit(x_train, y_train, epochs=nepochs, batch_size=128, 
                    shuffle=True, verbose=1, validation_data=(x_test,y_test))
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
    trainer = AMSGradPlus(lr=0.001, mu=0.7)
    model = create_model()
    model.compile(optimizer=trainer, loss=keras.losses.categorical_crossentropy, 
                    metrics=['accuracy'])
    log = model.fit(x_train, y_train, epochs=nepochs, batch_size=128, 
                    shuffle=True, verbose=1, validation_data=(x_test,y_test))
    keras.backend.clear_session()
    loss.append(log.history['loss'])
    # val.append(log.history['val_loss'])
loss = np.array(loss)
# val = np.array(val)
np.savetxt(directory+'AMSGrad+_loss.txt',loss,fmt = '%.8f')
# np.savetxt(directory+'AMSGrad+_val.txt',val,fmt = '%.8f')