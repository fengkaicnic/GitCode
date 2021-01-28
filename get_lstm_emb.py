import tensorflow as tf
import keras
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

kernel = [[-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1]]
kernel = np.array((kernel), dtype="float32")
rrate = 3

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
    vdname = '_'.join(name.split('_')[:3])
    num = int(name.split('.')[0].split('_')[3])
    nn = '_'.join(name.split('.')[0].split('_')[:3])
    if num < capnum-lstmnum:
        try:
            for i in range(lstmnum):
                imgname = nn + '_' + str(num + i) + '.jpg'
                img = cv2.imread(fake_path + imgname)
                img = cv2.resize(img, (128, 100))
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


#model = load_model('weights-improvement-27-0.79.hdf5')
model.load_weights('weights-m0-03-0.78.hdf5')

flat_layer = Model(model.input, outputs=model.get_layer('flatten_1').output)

test_out = flat_layer.predict(testx)
train_out = flat_layer.predict(trainx)
print(len(test_out))
pdb.set_trace()

trainyy = trainy[:, 0]
testyy = testy[:, 0]

import catboost as ctb
from catboost import CatBoostClassifier, CatBoostRegressor
metricname = 'Accuracy'
#model = CatBoostClassifier(iterations=10000, depth=3, bagging_temperature=0.2, l2_leaf_reg=50,
#                            custom_metric=metricname, learning_rate=0.5, eval_metric=metricname, loss_function='Logloss',
#                            logging_level='Verbose')

model = CatBoostRegressor(iterations=10000, depth=3, bagging_temperature=0.2, l2_leaf_reg=50,
                            custom_metric=metricname, learning_rate=0.5, eval_metric=metricname, loss_function='Logloss',
                            logging_level='Verbose')

model.fit(train_out, trainyy,eval_set=(test_out, testyy), plot=False)
pdb.set_trace()
predict = model.predict_proba(test_out)

np.save('predictcat', predict)
np.save('testcat', testyy)

