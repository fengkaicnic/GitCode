import tensorflow as tf
from keras import Sequential
from keras.backend import categorical_crossentropy
from keras.layers import ConvLSTM2D, Flatten, Dense, BatchNormalization, MaxPool2D, MaxPool3D

import cv2
import numpy as np
import os
import random
import time
import pdb

from keras.optimizers import Adadelta
import keras
import pdb

st = time.time()

data = []
y = []
trainx = []
trainy = []
testx = []
testy = []
real_path = '/mnt/celeb-real-lstm/'
fake_path = '/mnt/celeb-synthesis-lstm/'
#pdb.set_trace()
lstmnum = 4
capnum = 7
alltotal = 900
total = alltotal
for name in os.listdir(real_path):
    dd = []
    if total <= 0:
        break
    num = int(name.split('.')[0].split('_')[2])
    nn = '_'.join(name.split('.')[0].split('_')[:2])
    if num < capnum-lstmnum:
        try:
            for i in range(lstmnum):
                imgname = nn+'_'+str(num+i)+'.jpg'
                img = cv2.imread(real_path+imgname)
                img = cv2.resize(img, (128, 100))
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

total = alltotal
for name in os.listdir(fake_path):
    dd = []
    if total <= 0:
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

        if total > alltotal * 0.2:
            trainx.append(dd)
            trainy.append([0, 1])
        else:
            testx.append(dd)
            testy.append([0, 1])
        total -= 1
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

print(len(trainx))
print(len(testx))
model = Sequential()
#pdb.set_trace()
# lstm_input = np.random.random((4, 6, 30, 30, 3)).astype(np.float32)
# lstm_input = tf.convert_to_tensor(trainx)
input1 = keras.layers.Input(shape=(lstmnum, 100, 128, 3))
lstm_out1 = ConvLSTM2D(filters=1, kernel_size=[3, 3], strides=(1, 1), padding='valid', activation='relu',
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
epochs = 50
# # pdb.set_trace()
model.fit(trainx, trainy,
         batch_size=batch_size,
         epochs=epochs, validation_data=(testx, testy))

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     lstm_out1_, lstm_out2_, lstm_out3_, flat_, = sess.run([lstm_out1, lstm_out2, lstm_out3, flat])
#     print(lstm_out1_.shape)
#     print(lstm_out2_.shape)
#     print(lstm_out3_.shape)
#     print(flat_.shape)
