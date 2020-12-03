import keras
import numpy as np
import matplotlib.pyplot as plt

import random

from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.optimizers import RMSprop
import cv2
from keras.layers import Conv2D, AvgPool2D, MaxPool2D
from keras.layers.normalization import BatchNormalization

num_classes = 10
epochs = 60


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def dense_layer(vects):
    x, y = vects
    import keras.backend as K

    return K.concatenate([x, y])


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)


def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)


def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)
    x = BatchNormalization()(input)
    x = Conv2D(32, (5, 5), activation='relu', kernel_initializer='random_uniform')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='random_uniform')(x)
    # x = MaxPool2D(pool_size=(2,2))(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                           gamma_initializer='ones', moving_mean_initializer='zeros',
                           moving_variance_initializer='ones')(x)
    x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='random_uniform')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                           gamma_initializer='ones', moving_mean_initializer='zeros',
                           moving_variance_initializer='ones')(x)
    x = Conv2D(16, (3, 3), activation='relu', kernel_initializer='random_uniform')(x)
    x = AvgPool2D(pool_size=(3, 3), strides=2)(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                           gamma_initializer='ones', moving_mean_initializer='zeros',
                           moving_variance_initializer='ones')(x)
    x = Conv2D(16, (3, 3), activation='relu', kernel_initializer='random_uniform')(x)
    x = AvgPool2D(pool_size=(3, 3), strides=2)(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                           gamma_initializer='ones', moving_mean_initializer='zeros',
                           moving_variance_initializer='ones')(x)
    x = Conv2D(16, (3, 3), activation='relu', kernel_initializer='random_uniform')(x)
    x = MaxPool2D(pool_size=(3, 3), strides=2)(x)
    # x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones')(x)

    x = Flatten()(x)
    return Model(input, x)


def compute_accuracy(y_true, y_pred): # numpy上的操作
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred): # Tensor上的操作
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

def plot_train_history(history, train_metrics, val_metrics):
    plt.plot(history.history.get(train_metrics), '-o')
    plt.plot(history.history.get(val_metrics), '-o')
    plt.ylabel(train_metrics)
    plt.xlabel('Epochs')
    plt.legend(['train', 'validation'])

def load_data(path, label):
    import os
    xt = []
    yt = []
    xtest = []
    ytest = []
    for name in os.listdir(path)[:80]:
        for i in range(1, 8):
            try:
                img1 = cv2.imread(path+name+'/'+str(i)+ '.jpg')
                img2 = cv2.imread(path+name+'/'+str(i+1)+'.jpg')
                img1 = cv2.resize(img1, (100, 128)).astype('float32') / 255
                img2 = cv2.resize(img2, (100, 128)).astype('float32') / 255
                xt.append([img1, img2])
                yt.append(label)
            except:
                # import pdb
                # pdb.set_trace()
                pass
    for name in os.listdir(path)[80:]:
        for i in range(1, 8):
            try:
                img1 = cv2.imread(path+name+'/'+str(i)+ '.jpg')
                img2 = cv2.imread(path+name+'/'+str(i+1)+'.jpg')
                img1 = cv2.resize(img1, (100, 128)).astype('float32') / 255
                img2 = cv2.resize(img2, (100, 128)).astype('float32') / 255
                xtest.append([img1, img2])
                ytest.append(label)
            except:
                # import pdb
                # pdb.set_trace()
                pass
    return xt, yt, xtest, ytest

# the data, split between train and test sets
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
x1, y1, xtest, ytest = load_data('E:/code/frame/Celeb-real_vcon/', 1)
x2, y2, x2test, y2test = load_data('E:/code/frame/Celeb-synthesis_vcon/', 0)
x_train , y_train , x_test, y_test = [], [], [], []
import pdb
# pdb.set_trace()
for i in range(len(x1)):
        try:
            x_train.append(x1[i])
            y_train.append(y1[i])
        except:
            pdb.set_trace()
for i in range(len(x2)):
        x_train.append(x2[i])
        y_train.append(y2[i])
for i in range(len(xtest)):
        x_test.append(xtest[i])
        y_test.append(ytest[i])
for i in range(len(x2test)):
        x_test.append(x2test[i])
        y_test.append(y2test[i])

import random

randnum = random.randint(0,100)
random.seed(randnum)
random.shuffle(x_train)
random.seed(randnum)
random.shuffle(y_train)
random.seed(randnum)
random.shuffle(x_test)
random.seed(randnum)
random.shuffle(y_test)
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
# pdb.set_trace()
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
input_shape = (128, 100, 3)

# create training+test positive and negative pairs
#digit_indices = [np.where(y_train == i)[0] for i in range(num_classes)]
# tr_pairs, tr_y = create_pairs(x_train, digit_indices)

#digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
# te_pairs, te_y = create_pairs(x_test, digit_indices)

# network definition
base_network = create_base_network(input_shape)

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

# distance = Lambda(euclidean_distance,
#                   output_shape=eucl_dist_output_shape)([processed_a, processed_b])

import keras.backend as K
import pdb
#pdb.set_trace()
# x_all = K.concatenate([processed_a, processed_b])
x_all = Lambda(dense_layer)([processed_a, processed_b])
out = Dense(1, activation='sigmoid')(x_all)

model = Model([input_a, input_b], out)
#keras.utils.plot_model(model,"siamModel.png",show_shapes=True)
model.summary()

# train
rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
# pdb.set_trace()
history=model.fit([x_train[:, 0], x_train[:, 1]], y_train,
          batch_size=64,
          epochs=epochs,verbose=2,
          validation_data=([x_test[:, 0], x_test[:, 1]], y_test))

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plot_train_history(history, 'loss', 'val_loss')
plt.subplot(1, 2, 2)
plot_train_history(history, 'accuracy', 'val_accuracy')
plt.show()


# compute final accuracy on training and test sets
y_pred = model.predict([x_train[:, 0], x_train[:, 1]])
tr_acc = compute_accuracy(y_train, y_pred)
y_pred = model.predict([x_test[:, 0], x_test[:, 1]])
te_acc = compute_accuracy(y_test, y_pred)

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
