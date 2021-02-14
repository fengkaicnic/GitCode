import tensorflow as tf
from keras.models import load_model
from keras.models import Model
import cv2
import pdb
import numpy as np
import os
import random
from sklearn import metrics

kernel = [[-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1]]
kernel = np.array((kernel), dtype="float32")
rrate = 1

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
alltotal = 80
total = alltotal
ttname = ''
vdname = ''
fflag = 1
pdb.set_trace()
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

pdb.set_trace()

def auroc(y_true, y_pred):
    return tf.py_func(metrics.roc_auc_score, (y_true, y_pred), tf.double)

model.load_weights('weights-improvement-27-0.79.hdf5')
pdb.set_trace()

flat_layer = Model(model.input, outputs=model.get_layer('flatten_1').output)

flat_out = flat_layer.predict(testx)
pdb.set_trace()
print(len(flat_out))

