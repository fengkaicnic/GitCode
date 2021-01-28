import tensorflow as tf
import keras 
from keras.utils import plot_model
from keras import Sequential, regularizers
from keras.backend import categorical_crossentropy
from keras.layers import ConvLSTM2D, Flatten, Dense, BatchNormalization, MaxPool2D, MaxPool3D
from keras.constraints import Constraint
from keras.constraints import max_norm
from keras.callbacks import ModelCheckpoint
from keras.layers import LeakyReLU, Dropout
from keras.models import load_model
from keras.models import Model
from keras.optimizers import Adadelta

import cv2
import pdb
import numpy as np
import os
import random
from sklearn import metrics
import createmodel



model = createmodel.create_m0()

def auroc(y_true, y_pred):
    return tf.py_func(metrics.roc_auc_score, (y_true, y_pred), tf.double)

model.compile(loss=categorical_crossentropy,
             optimizer=Adadelta(),
             metrics=['accuracy', auroc])

batch_size = 32
epochs = 60
# pdb.set_trace()
filepath = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                            mode='max')
callbacks_list = [checkpoint]


#model = load_model('weights-m0-03-0.78.hdf5')
model.load_weights('weights-m0-03-0.78.hdf5')
pdb.set_trace()
print(model.count_params())
print(model.summary())
#plot_model(model, to_file='model.png')

