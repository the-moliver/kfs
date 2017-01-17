'''Trains a simple deep NN on the MNIST dataset.

'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils
from kfs.layers import FilterDims
from kfs.layers.noise import CoupledGaussianDropout
from kfs.constraints import Stochastic
import matplotlib.pyplot as plt

batch_size = 128
nb_classes = 10
nb_epoch = 20

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(1, 60000, 784)
X_train = X_train.astype('float32')
X_train /= 255

print(X_train.shape[0], 'train samples')


def mean_abs_error(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true))

model = Sequential()
model.add(FilterDims(nb_filters=20, sum_axes=[1], filter_axes=[1], input_shape=(60000, 784,), bias=False, W_constraint=Stochastic(axis=0)))
model.add(CoupledGaussianDropout())
model.add(FilterDims(nb_filters=60000, sum_axes=[1], filter_axes=[1], bias=False, W_constraint=Stochastic(axis=0)))

model.summary()

model.compile(loss=mean_abs_error,
              optimizer=Adam())

history = model.fit(X_train, X_train,
                    batch_size=batch_size, nb_epoch=100,
                    verbose=1)

W=model.get_weights()
A=np.sum(X_train.reshape(60000,784, 1) * W[0], axis=0)

def show_archetype(component, W, A, X_train, num_examples=4):
    idx = W[0][:,0,component].argsort()[::-1]
    plt.figure(1)
    plt.subplot(1,num_examples+1,1)
    plt.imshow(A[:,component].reshape(28,28), cmap='viridis')
    plt.axis('off')
    plt.title('Archetype')

    for i in range(num_examples):
        plt.subplot(1,num_examples+1,i+2)
        plt.imshow(X_train[0,idx[i],:].reshape(28,28), cmap='viridis')
        plt.axis('off')
        plt.title('Weight: %3.2f' % W[0][idx[i],0,component])

show_archetype(12, W, A, X_train)