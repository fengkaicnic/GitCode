import tensorflow as tf
from keras import Sequential, regularizers
from keras.backend import categorical_crossentropy
from keras.layers import ConvLSTM2D, Flatten, Dense, BatchNormalization, MaxPool2D, MaxPool3D
from keras.constraints import Constraint
from keras.constraints import max_norm
from keras.callbacks import ModelCheckpoint
from keras.layers import LeakyReLU, Dropout
from keras.models import load_model

import cv2
import numpy as np
import os
import random
import time
from sklearn import metrics
import sys

from keras.optimizers import Adadelta
import keras
import pdb

st = time.time()
rrate = 3
class BayarConstraint( Constraint ) :
    def __init__( self ) :
        self.mask = None
    def _initialize_mask( self, w ) :
        nb_rows, nb_cols, nb_inputs, nb_outputs = K.int_shape(w)
        m = np.zeros([nb_rows, nb_cols, nb_inputs, nb_outputs]).astype('float32')
        m[nb_rows//2,nb_cols//2] = 1.
        self.mask = K.variable( m, dtype='float32' )
        return
    def __call__( self, w ) :
        if self.mask is None :
            self._initialize_mask(w)
        w *= (1-self.mask)
        rest_sum = K.sum( w, axis=(0,1), keepdims=True)
        w /= rest_sum + K.epsilon()
        w -= self.mask
        return w

import generate_data
trainx, trainy, testx, testy = generate_data.generate()

import createmodel

model = createmodel.create_m64()

def auroc(y_true, y_pred):
    return tf.py_func(metrics.roc_auc_score, (y_true, y_pred), tf.double)

model.compile(loss=categorical_crossentropy,
             optimizer=Adadelta(),
             metrics=['accuracy'])

batch_size = 32
epochs = 60
# pdb.set_trace()
filepath = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                            mode='max')
callbacks_list = [checkpoint]
modelpath = sys.argv[1]
model.load_weights(modelpath)
#pdb.set_trace()
history = model.fit(trainx, trainy,
         batch_size=batch_size,callbacks=callbacks_list,
         epochs=epochs, validation_data=(testx, testy))

#pdb.set_trace()
predictx = model.predict(testx)
#model.save('model_weight.h5')
#print(predictx)
np.save("testy"+str(rrate), testy)
np.save("predictx"+str(rrate), predictx)
t_auc = metrics.roc_auc_score(testy, predictx)
print(t_auc)
print(predictx.shape, testy.shape)
print(podata,negdata)
#np.save("testy3", testy)
#np.save("predictx3", predictx)

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     lstm_out1_, lstm_out2_, lstm_out3_, flat_, = sess.run([lstm_out1, lstm_out2, lstm_out3, flat])
#     print(lstm_out1_.shape)
#     print(lstm_out2_.shape)
#     print(lstm_out3_.shape)
#     print(flat_.shape)
