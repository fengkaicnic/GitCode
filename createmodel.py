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

def create_m0():
    model = Sequential()
    lstmnum= 4

    input1 = keras.layers.Input(shape=(lstmnum, 100, 128, 3))
    lstm_out1 = ConvLSTM2D(filters=1, kernel_size=[3, 3], strides=(1, 1), padding='valid', kernel_constraint=max_norm(2.), activation='relu',
                       input_shape=(lstmnum, 100, 128, 3), return_sequences=True)(input1)
    x = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                           gamma_initializer='ones', moving_mean_initializer='zeros',
                           moving_variance_initializer='ones')(lstm_out1)
    x = MaxPool3D(pool_size=(2, 2, 2))(x)
    x = ConvLSTM2D(filters=3, kernel_size=[3, 3], strides=(1, 1), padding='valid', activation='relu',
                       return_sequences=True)(x)
    x = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                           gamma_initializer='ones', moving_mean_initializer='zeros',
                           moving_variance_initializer='ones')(x)
    x = MaxPool3D(pool_size=(2, 2, 2))(x)
    flat = Flatten()(x)
    out = Dropout(0.3)(flat)
    out = Dense(2, kernel_regularizer=regularizers.l2(0.01), activation='softmax')(out)

    model = keras.models.Model(inputs=input1, outputs=out)
    return model


def create_m1():
    model = Sequential()
    lstmnum= 4

    input1 = keras.layers.Input(shape=(lstmnum, 100, 128, 3))
    lstm_out1 = ConvLSTM2D(filters=1, kernel_size=[3, 3], strides=(1, 1), padding='valid', kernel_constraint=max_norm(2.), activation='relu',
                       input_shape=(lstmnum, 100, 128, 3), return_sequences=True)(input1)
    x = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                           gamma_initializer='ones', moving_mean_initializer='zeros',
                           moving_variance_initializer='ones')(lstm_out1)
    x = MaxPool3D(pool_size=(2, 2, 2))(x)
    x = ConvLSTM2D(filters=3, kernel_size=[3, 3], strides=(1, 1), padding='valid', activation='relu',
                       return_sequences=True)(x)
    x = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                           gamma_initializer='ones', moving_mean_initializer='zeros',
                           moving_variance_initializer='ones')(x)
    x = MaxPool3D(pool_size=(2, 2, 2))(x)
    flat = Flatten()(x)
    out = Dropout(0.3)(flat)
    out = Dense(2, kernel_regularizer=regularizers.l2(0.01), activation='softmax')(out)

    model = keras.models.Model(inputs=input1, outputs=out)
    return model

lstmnum = 4
def create_m2():
    model = Sequential()
    lstmnum = 4
    input1 = keras.layers.Input(shape=(lstmnum, 100, 128, 3))
    lstm_out1 = ConvLSTM2D(filters=1, kernel_size=[3, 3], strides=(1, 1), padding='valid', kernel_constraint=max_norm(2.), activation='relu',
                       input_shape=(lstmnum, 100, 128, 3), return_sequences=True)(input1)
    x = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                           gamma_initializer='ones', moving_mean_initializer='zeros',
                           moving_variance_initializer='ones')(lstm_out1)
    x = MaxPool3D(pool_size=(2, 2, 2))(x)
    lstm_out2 = ConvLSTM2D(filters=2, kernel_size=[3, 3], strides=(1, 1), padding='valid', activation='relu',
                       return_sequences=True)(x)
    x = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                           gamma_initializer='ones', moving_mean_initializer='zeros',
                           moving_variance_initializer='ones')(x)
    x = MaxPool3D(pool_size=(2, 2, 2))(x)
    x = ConvLSTM2D(filters=3, kernel_size=[3, 3], strides=(1, 1), padding='valid', activation='relu',
                       return_sequences=True)(x)
    x = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                           gamma_initializer='ones', moving_mean_initializer='zeros',
                           moving_variance_initializer='ones')(x)
    flat = Flatten()(x)
    out = Dropout(0.3)(flat)
    out = Dense(2, kernel_regularizer=regularizers.l2(0.01), activation='softmax')(out)
    model = keras.models.Model(inputs=input1, outputs=out)
    return model

def create_m3():
    lstmnum = 4
    model = Sequential()
    input1 = keras.layers.Input(shape=(lstmnum, 100, 128, 3))
    lstm_out1 = ConvLSTM2D(filters=1, kernel_size=[3, 3], strides=(1, 1), padding='valid', kernel_constraint=max_norm(2.), activation='relu',
                       input_shape=(lstmnum, 100, 128, 3), return_sequences=True)(input1)
    x = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                           gamma_initializer='ones', moving_mean_initializer='zeros',
                           moving_variance_initializer='ones')(lstm_out1)
    x = MaxPool3D(pool_size=(2, 2, 2))(x)
    lstm_out2 = ConvLSTM2D(filters=2, kernel_size=[3, 3], strides=(1, 1), padding='valid', activation='relu',
                       return_sequences=True)(x)
    x = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                           gamma_initializer='ones', moving_mean_initializer='zeros',
                           moving_variance_initializer='ones')(lstm_out2)
    x = MaxPool3D(pool_size=(2, 2, 2))(x)
    x = ConvLSTM2D(filters=3, kernel_size=[3, 3], strides=(1, 1), padding='valid', activation='relu',
                       return_sequences=True)(x)
    x = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                           gamma_initializer='ones', moving_mean_initializer='zeros',
                           moving_variance_initializer='ones')(x)
    #x = MaxPool3D(pool_size=(2, 2, 2))(x)
    x = ConvLSTM2D(filters=3, kernel_size=[3, 3], strides=(1, 1), padding='valid', activation='relu',
                       return_sequences=True)(x)
    x = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                           gamma_initializer='ones', moving_mean_initializer='zeros',
                           moving_variance_initializer='ones')(x)
    flat = Flatten()(x)
    out = Dropout(0.3)(flat)
    out = Dense(2, kernel_regularizer=regularizers.l2(0.01), activation='softmax')(out)
    model = keras.models.Model(inputs=input1, outputs=out)
    return model

def create_m4():
    lstmnum = 4
    model = Sequential()
    input1 = keras.layers.Input(shape=(lstmnum, 100, 128, 3))
    lstm_out1 = ConvLSTM2D(filters=1, kernel_size=[3, 3], strides=(1, 1), padding='valid', kernel_constraint=max_norm(2.), activation='relu',
                       input_shape=(lstmnum, 100, 128, 3), return_sequences=True)(input1)
    x = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                           gamma_initializer='ones', moving_mean_initializer='zeros',
                           moving_variance_initializer='ones')(lstm_out1)
    x = MaxPool3D(pool_size=(2, 2, 2))(x)
    lstm_out2 = ConvLSTM2D(filters=2, kernel_size=[3, 3], strides=(1, 1), padding='valid', activation='relu',
                       return_sequences=True)(x)
    x = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                           gamma_initializer='ones', moving_mean_initializer='zeros',
                           moving_variance_initializer='ones')(lstm_out2)
    x = MaxPool3D(pool_size=(2, 2, 2))(x)
    x = ConvLSTM2D(filters=3, kernel_size=[3, 3], strides=(1, 1), padding='valid', activation='relu',
                       return_sequences=True)(x)
    x = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                           gamma_initializer='ones', moving_mean_initializer='zeros',
                           moving_variance_initializer='ones')(x)
    #x = MaxPool3D(pool_size=(2, 2, 2))(x)
    x = ConvLSTM2D(filters=3, kernel_size=[3, 3], strides=(1, 1), padding='valid', activation='relu',
                       return_sequences=True)(x)
    x = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                           gamma_initializer='ones', moving_mean_initializer='zeros',
                           moving_variance_initializer='ones')(x)
    x = ConvLSTM2D(filters=3, kernel_size=[3, 3], strides=(1, 1), padding='valid', activation='relu',
                       return_sequences=True)(x)
    x = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                           gamma_initializer='ones', moving_mean_initializer='zeros',
                           moving_variance_initializer='ones')(x)
    flat = Flatten()(x)
    out = Dropout(0.3)(flat)
    out = Dense(2, kernel_regularizer=regularizers.l2(0.01), activation='softmax')(out)
    model = keras.models.Model(inputs=input1, outputs=out)
    return model

def create_m4_old():
    model = Sequential()
    input1 = keras.layers.Input(shape=(lstmnum, 100, 128, 3))
    lstm_out1 = ConvLSTM2D(filters=1, kernel_size=[3, 3], strides=(1, 1), padding='valid', kernel_constraint=max_norm(2.), activation='relu',
                       input_shape=(lstmnum, 100, 128, 3), return_sequences=True)(input1)
    x = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                           gamma_initializer='ones', moving_mean_initializer='zeros',
                           moving_variance_initializer='ones')(lstm_out1)
    x = MaxPool3D(pool_size=(2, 2, 2))(x)
    lstm_out2 = ConvLSTM2D(filters=2, kernel_size=[3, 3], strides=(1, 1), padding='valid', activation='relu',
                       return_sequences=True)(x)
    x = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                           gamma_initializer='ones', moving_mean_initializer='zeros',
                           moving_variance_initializer='ones')(lstm_out2)
    x = MaxPool3D(pool_size=(2, 2, 2))(x)
    x = ConvLSTM2D(filters=3, kernel_size=[3, 3], strides=(1, 1), padding='valid', activation='relu',
                       return_sequences=True)(x)
    x = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                           gamma_initializer='ones', moving_mean_initializer='zeros',
                           moving_variance_initializer='ones')(lstm_out2)
    x = MaxPool3D(pool_size=(2, 2, 2))(x)
    x = ConvLSTM2D(filters=3, kernel_size=[3, 3], strides=(1, 1), padding='valid', activation='relu',
                       return_sequences=True)(x)
    x = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                           gamma_initializer='ones', moving_mean_initializer='zeros',
                           moving_variance_initializer='ones')(lstm_out2)
    x = MaxPool3D(pool_size=(2, 2, 2))(x)
    x = ConvLSTM2D(filters=3, kernel_size=[3, 3], strides=(1, 1), padding='valid', activation='relu',
                       return_sequences=True)(x)
    x = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                           gamma_initializer='ones', moving_mean_initializer='zeros',
                           moving_variance_initializer='ones')(lstm_out2)
    x = MaxPool3D(pool_size=(2, 2, 2))(x)
    flat = Flatten()(x)
    out = Dropout(0.3)(flat)
    out = Dense(2, kernel_regularizer=regularizers.l2(0.01), activation='softmax')(out)
    model = keras.models.Model(inputs=input1, outputs=out)
    return model

def create_m64():
    model = Sequential()
    input1 = keras.layers.Input(shape=(lstmnum, 100, 128, 3))
    lstm_out1 = ConvLSTM2D(filters=1, kernel_size=[3, 3], strides=(1, 1), padding='valid', kernel_constraint=max_norm(2.), activation='relu',
                       input_shape=(lstmnum, 100, 128, 3), return_sequences=True)(input1)
    x = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                           gamma_initializer='ones', moving_mean_initializer='zeros',
                           moving_variance_initializer='ones')(lstm_out1)
    x = MaxPool3D(pool_size=(2, 2, 2))(x)
    lstm_out2 = ConvLSTM2D(filters=2, kernel_size=[3, 3], strides=(1, 1), padding='valid', activation='relu',
                       return_sequences=True)(x)
    x = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                           gamma_initializer='ones', moving_mean_initializer='zeros',
                           moving_variance_initializer='ones')(lstm_out2)
    x = MaxPool3D(pool_size=(2, 2, 2))(x)
    x = ConvLSTM2D(filters=3, kernel_size=[3, 3], strides=(1, 1), padding='valid', activation='relu',
                       return_sequences=True)(x)
    x = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                           gamma_initializer='ones', moving_mean_initializer='zeros',
                           moving_variance_initializer='ones')(lstm_out2)
    x = MaxPool3D(pool_size=(2, 2, 2))(x)
    x = ConvLSTM2D(filters=3, kernel_size=[3, 3], strides=(1, 1), padding='valid', activation='relu',
                       return_sequences=True)(x)
    x = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                           gamma_initializer='ones', moving_mean_initializer='zeros',
                           moving_variance_initializer='ones')(lstm_out2)
    x = MaxPool3D(pool_size=(2, 2, 2))(x)
    x = ConvLSTM2D(filters=3, kernel_size=[3, 3], strides=(1, 1), padding='valid', activation='relu',
                       return_sequences=True)(x)
    x = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                           gamma_initializer='ones', moving_mean_initializer='zeros',
                           moving_variance_initializer='ones')(lstm_out2)
    x = MaxPool3D(pool_size=(2, 2, 2))(x)
    flat = Flatten()(x)
    out = Dropout(0.3)(flat)
    out = Dense(124, kernel_regularizer=regularizers.l2(0.01), activation='softmax')(out)
    model = keras.models.Model(inputs=input1, outputs=out)
    return model

