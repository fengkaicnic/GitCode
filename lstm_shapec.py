import tensorflow as tf
from keras import Sequential, regularizers
from keras.backend import categorical_crossentropy, binary_crossentropy
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

kernel = [[-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1]]
kernel = np.array((kernel), dtype="float32")

data = []
y = []
trainx = []
trainy = []
testx = []
testy = []
real_path = '/mnt/celeb-real-lstm/'
#fake_path = '/mnt/celeb-synthesis-lstm/'
#real_path = '/mnt/celeb-real-eye/'
fake_path = '/mnt/celeb-synthesis-eye/'
#pdb.set_trace()
lstmnum = 4
capnum = 7
alltotal = 800
total = alltotal
ttname = ''
vdname = ''
fflag = 1
for name in os.listdir(real_path):
    dd = []
    if total <= 0:
        break
    filterr = 1
    if np.random.randint(100) < 92:
        filterr = 1
    #print(name)
    vdname = '_'.join(name.split('_')[:2])
    num = int(name.split('.')[0].split('_')[2])
    nn = '_'.join(name.split('.')[0].split('_')[:2])
    if num < capnum-lstmnum:
        try:
            for i in range(lstmnum):
                imgname = nn+'_'+str(num+i)+'.jpg'
                img = cv2.imread(real_path+imgname)
                img = cv2.resize(img, (128, 100))
                if filterr:
                    img = cv2.filter2D(img, -1, kernel)
                dd.append(img)
            dd = np.array(dd)
        except:
            print(name)
            continue

        if total > alltotal * 0.2:
            trainx.append(dd)
            trainy.append([1, 0])
        else:
            if vdname == 'ttname' and fflag:
                continue
            fflag = 0
            testx.append(dd)
            testy.append([1, 0])
        total -= 1
        ttname = vdname
#pdb.set_trace()
print(len(trainx), len(testx))
podata = len(testy)
total = alltotal*rrate
ftotal = alltotal*rrate
fflag = 1
for name in os.listdir(fake_path):
    #if np.random.randint(2) == 1:
    #    continue
    dd = []
    if ftotal <= 0:
        break
    filterr = 1
    if np.random.randint(100) < 92:
        filterr = 1
    vdname = '_'.join(name.split('_')[:3])
    num = int(name.split('.')[0].split('_')[3])
    nn = '_'.join(name.split('.')[0].split('_')[:3])
    if num < capnum-lstmnum:
        try:
            for i in range(lstmnum):
                imgname = nn + '_' + str(num + i) + '.jpg'
                img = cv2.imread(fake_path + imgname)
                img = cv2.resize(img, (128, 100))
                if filterr:
                    img = cv2.filter2D(img, -1, kernel)
                dd.append(img)
            dd = np.array(dd)
        except:
            print(name)
            continue

        if ftotal > total * 0.2:
            trainx.append(dd)
            trainy.append([0, 1])
        else:
            if ttname == vdname and fflag:
                continue
            fflag = 0
            testx.append(dd)
            testy.append([0, 1])
        ftotal -= 1
        ttname = vdname
negdata = len(testy) - podata
#pdb.set_trace()

seed = random.randint(0, 100)
random.seed(seed)
random.shuffle(trainx)
random.seed(seed)
random.shuffle(trainy)
random.seed(seed)
random.shuffle(testx)
random.seed(seed)
random.shuffle(testy)

#pdb.set_trace()
trainx = np.array(trainx)
trainy = np.array(trainy)
trainx = trainx.reshape(-1, lstmnum, 100, 128, 3)
trainx = trainx.astype('float32')
testx = np.array(testx)
testx = testx.reshape(-1, lstmnum, 100, 128, 3)
testx = testx.astype('float32')
testy = np.array(testy)
#pdb.set_trace()
print(len(trainx))
print(len(testx))
print(podata, negdata)

import createmodel

model = createmodel.create_m0()

def auroc(y_true, y_pred):
    return tf.py_func(metrics.roc_auc_score, (y_true, y_pred), tf.double)

#categorical_crossentropy
model.compile(loss=binary_crossentropy,
             optimizer=Adadelta(),
             metrics=['accuracy', auroc])

batch_size = 32
epochs = 60
# pdb.set_trace()
filepath = "weights-m0-{epoch:02d}-{val_acc:.2f}.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                            mode='max')
callbacks_list = [checkpoint]
#model.load_weights('weights-improvement-27-0.79.hdf5')
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
