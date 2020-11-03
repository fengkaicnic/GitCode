import tensorflow as tf
from keras import Sequential
from keras.backend import categorical_crossentropy
from keras.layers import ConvLSTM2D, Flatten, Dense, BatchNormalization, MaxPool2D, MaxPool3D, Conv2D, Dropout, \
    AvgPool2D

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
real_path = 'E:/deepfake/Celeb-DF/lstmcap/real/'
fake_path = 'E:/deepfake/Celeb-DF/lstmcap/fake/'
#pdb.set_trace()
lstmnum = 1
capnum = 18
alltotal = 1200
total = alltotal
for name in os.listdir(real_path):
    dd = []
    if total <= 0:
        break
    num = int(name.split('.')[0].split('_')[2])
    if num > 10:
        continue
    try:
        img = cv2.imread(real_path+name)
        img = cv2.resize(img, (128, 100))
    except:
        print(name)
        continue

    if total > alltotal * 0.2:
        trainx.append(img)
        trainy.append([1, 0])
    else:
        testx.append(img)
        testy.append([1, 0])
    total -= 1

total = alltotal
for name in os.listdir(fake_path):
    dd = []
    if total <= 0:
        break
    # pdb.set_trace()
    num = int(name.split('.')[0].split('_')[3])
    if num > 10:
        continue
    try:
        img = cv2.imread(fake_path + name)
        img = cv2.resize(img, (128, 100))
    except:
        print(name)
        continue

    if total > alltotal * 0.2:
        trainx.append(img)
        trainy.append([0, 1])
    else:
        testx.append(img)
        testy.append([0, 1])
    total -= 1
# pdb.set_trace()

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
trainx = trainx.reshape(-1, 100, 128, 3)
trainx = trainx.astype('float32')
testx = np.array(testx)
testx = testx.reshape(-1, 100, 128, 3)
testx = testx.astype('float32')
testy = np.array(testy)

print(len(trainx))
print(len(testx))
pdb.set_trace()
model = Sequential()
model.add(Conv2D(8, (3,3), activation='relu',kernel_initializer='random_uniform', input_shape=(100, 128, 3)))
# model.add(AvgPool2D(pool_size=(2,2)))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'))
model.add(AvgPool2D(pool_size=(2,2)))
model.add(Conv2D(8, (3,3), activation='relu', kernel_initializer='random_uniform'))
# model.add(AvgPool2D(pool_size=(2,2)))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'))
model.add(AvgPool2D(pool_size=(2,2)))
model.add(Conv2D(16, (3,3), activation='relu', kernel_initializer='random_uniform'))
# model.add(AvgPool2D(pool_size=(2,2)))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'))
model.add(AvgPool2D(pool_size=(2,2)))
model.add(Conv2D(16, (3,3), activation='relu', kernel_initializer='random_uniform'))
# model.add(AvgPool2D(pool_size=(2,2)))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'))
model.add(AvgPool2D(pool_size=(2,2)))
model.add(Conv2D(32, (3,3), activation='relu', kernel_initializer='random_uniform'))
# model.add(AvgPool2D(pool_size=(2,2)))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'))
model.add(AvgPool2D(pool_size=(2,2)))
# model.add(MaxPool2D(pool_size=(2,2)))
# model.add(Conv2D(128, (3,3), activation='sigmoid', kernel_initializer='random_uniform'))
model.add(Flatten())
# model.add(Dropout(0.2))
# model.add(Dense(32, activation='sigmoid', kernel_initializer='random_uniform'))
model.add(Dropout(0.3))
model.add(Dense(2, activation='softmax'))

model.compile(loss=categorical_crossentropy,
             optimizer=Adadelta(),
             metrics=['accuracy'])

batch_size = 128
epochs = 96
# pdb.set_trace()
model.fit(trainx, trainy,
         batch_size=batch_size,
         epochs=epochs, validation_data=(testx, testy))

loss, accuracy = model.evaluate(trainx, trainy, verbose=1)
print('loss:%.4f accuracy:%.4f' %(loss, accuracy))

loss, accuracy = model.evaluate(testx, testy, verbose=1)
print('loss:%.4f accuracy:%.4f' %(loss, accuracy))
