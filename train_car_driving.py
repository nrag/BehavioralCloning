import math
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import time

from imagenet_utils import preprocess_input
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.layers import Flatten, Dense, Input
from keras.models import Model
from keras.optimizers import Adam
from scipy import misc 
from sklearn.model_selection import train_test_split
from vgg16 import VGG16

learning_rate = .00006
batch_size=32
epochs = 20 

#Load the data
image_data = pd.read_csv('../driving_log.csv', header=None)
y = image_data[3].values
image_paths = image_data[0].values

#Split the data into training and validation
train_paths, val_paths, y_train, y_validation = train_test_split(image_paths, y, test_size=0.33)
epoch_samples = (len(train_paths)//batch_size) * batch_size 
val_epoch_samples = (len(val_paths)//batch_size) * batch_size

print("Epoch Samples %d" % epoch_samples)
print("Validation Epoch Samples %d" % val_epoch_samples)

#Create generators for data
def batch_generator(paths, y, count):
    while 1:
        #print(len(paths))
        batch_index = np.random.random_integers(0, len(paths) - 1, count)
        #print(batch_index)
        batch_paths = paths[batch_index]
        #print(batch_paths)
        batch_X = None
        batch_y = y[batch_index]
        
        for i in range(len(batch_paths)):
            #print(p)
            p = batch_paths[i]
            image = misc.imread(p)
            rescaled = misc.imresize(image, (224,224,3))
            rescaled = rescaled.astype(np.float32)
            if np.random.random_sample() < 0.5:
                rescaled = np.fliplr(rescaled)
                batch_y[i] *= -1

            if (batch_X is None):
                batch_X = np.array([rescaled])
            else:
                batch_X = np.vstack((batch_X, [rescaled]))

        yield batch_X, batch_y

#Training data generator
def train_generator():
    return batch_generator(train_paths, y_train, batch_size)

#Validation data generator
def val_generator():
    return batch_generator(val_paths, y_validation, batch_size)

def step_decay(epoch):
    initial_lrate = learning_rate 
    drop = 0.5
    epochs_drop = 8.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

    
# Load the VGG model
model = VGG16(include_top=True, regularizer_lambda=0.0001, weights=None)

# Compile Model
model.compile(loss='mean_squared_error',
              optimizer=Adam(lr=learning_rate),
              metrics=['mean_squared_error'])

# Learning schedule callback
lrate = LearningRateScheduler(step_decay)

# Checkpoint the model
filepath="driving-model--{epoch:02d}-{val_loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

callbacks_list = [lrate, checkpoint]

# Fit the model
history = model.fit_generator(train_generator(), samples_per_epoch=epoch_samples, nb_epoch=epochs, callbacks=callbacks_list, validation_data=val_generator(), nb_val_samples=val_epoch_samples)

model.save('driving_model.h5')
