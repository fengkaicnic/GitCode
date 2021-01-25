import tensorflow as tf
from keras import Sequential
from keras.backend import categorical_crossentropy
from keras.layers import ConvLSTM2D, Flatten, Dense, BatchNormalization, MaxPool2D, MaxPool3D
from keras.constraints import Constraint
from keras.constraints import max_norm

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
rrate = 10
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
#real_path = '/mnt/celeb-real-lstm/'
#fake_path = '/mnt/celeb-synthesis-lstm/'
real_path = '/mnt/celeb-real-eye/'
fake_path = '/mnt/celeb-synthesis-eye/'
#pdb.set_trace()
lstmnum = 4
capnum = 7
alltotal = 364
total = alltotal
for name in os.listdir(real_path):
    dd = []
    if total <= 0:
        break
    print(name)
    num = int(name.split('.')[0].split('_')[2])
    nn = '_'.join(name.split('.')[0].split('_')[:2])
    if num < capnum-lstmnum:
        try:
            for i in range(lstmnum):
                imgname = nn+'_'+str(num+i)+'.jpg'
                img = cv2.imread(real_path+imgname)
                img = cv2.resize(img, (128, 100))
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
            testx.append(dd)
            testy.append([1, 0])
        total -= 1
pdb.set_trace()
podata = len(testy)
total = alltotal*rrate
ftotal = alltotal*rrate
for name in os.listdir(fake_path):
    if np.random.randint(2) == 1:
        continue
    dd = []
    if ftotal <= 0:
        break

    num = int(name.split('.')[0].split('_')[3])
    nn = '_'.join(name.split('.')[0].split('_')[:3])
    if num < capnum-lstmnum:
        try:
            for i in range(lstmnum):
                imgname = nn + '_' + str(num + i) + '.jpg'
                img = cv2.imread(fake_path + imgname)
                img = cv2.resize(img, (128, 100))
                dd.append(img)
            dd = np.array(dd)
        except:
            print(name)
            continue

        if ftotal > total * 0.2:
            trainx.append(dd)
            trainy.append([0, 1])
        else:
            testx.append(dd)
            testy.append([0, 1])
        ftotal -= 1
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
model = Sequential()
#pdb.set_trace()
# lstm_input = np.random.random((4, 6, 30, 30, 3)).astype(np.float32)
# lstm_input = tf.convert_to_tensor(trainx)
input1 = keras.layers.Input(shape=(lstmnum, 100, 128, 3))
lstm_out1 = ConvLSTM2D(filters=1, kernel_size=[3, 3], strides=(1, 1), padding='valid', kernel_constraint=max_norm(2.), activation='relu',
                       input_shape=(lstmnum, 100, 128, 3), return_sequences=True)(input1)
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                           gamma_initializer='ones', moving_mean_initializer='zeros',
                           moving_variance_initializer='ones')(lstm_out1)
x = MaxPool3D(pool_size=(2, 2, 2))(x)
lstm_out2 = ConvLSTM2D(filters=2, kernel_size=[3, 3], strides=(1, 1), padding='valid', activation='relu',
                       return_sequences=True)(x)
lstm_out3 = ConvLSTM2D(filters=3, kernel_size=[3, 3], strides=(1, 1), padding='valid', activation='relu',
                       return_sequences=True)(lstm_out2)
flat = Flatten()(x)
out = Dense(2, activation='softmax')(flat)

model = keras.models.Model(inputs=input1, outputs=out)

model.compile(loss=categorical_crossentropy,
             optimizer=Adadelta(),
             metrics=['accuracy'])

batch_size = 32
epochs = 6
# # pdb.set_trace()
model.fit(trainx, trainy,
         batch_size=batch_size,
         epochs=epochs, validation_data=(testx, testy))

pdb.set_trace()
predictx = model.predict(testx)
model.save('model_weight.h5')
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
