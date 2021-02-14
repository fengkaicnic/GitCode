import tensorflow as tf
from keras import Sequential, regularizers
from keras.backend import categorical_crossentropy
from keras.layers import ConvLSTM2D, Flatten, Dense, BatchNormalization, MaxPool2D, MaxPool3D
from keras.constraints import Constraint
from keras.constraints import max_norm
from keras.callbacks import ModelCheckpoint
from keras.layers import LeakyReLU, Dropout
from keras.models import load_model
from sklearn.metrics import roc_curve, auc
from scipy import interp

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

import generate_data
trainx, trainy, testx, testy = generate_data.generate()

import createmodel

model = createmodel.create_m64()

modelpath = sys.argv[1]
if modelpath != 'none':
    model.load_weights(modelpath)

import utils

poslabel = utils.translabel(trainy)

predictx = model.predict(trainx)
prelabel = utils.translabel(predixtx)

t_auc = metrics.roc_auc_score(poslabel, prelabel)
print(t_auc)

fpr, tpr, thresholds = roc_curve(poslabel[test], prelabel[:, 1])
tprs.append(interp(mean_fpr, fpr, tpr))
tprs[-1][0] = 0.0
roc_auc = auc(fpr, tpr)
aucs.append(roc_auc)
plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
