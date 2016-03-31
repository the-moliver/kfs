from __future__ import absolute_import
from keras import backend as K


def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)


def cosh(x):
    return .5*(K.exp(x) + K.exp(-x))


def half_squared_error(y_true, y_pred):
    return K.mean(K.log(cosh(K.maximum(0., y_pred - y_true))) + 0.5*K.square(K.maximum(0., y_true - y_pred)), axis=-1)

from keras.utils.generic_utils import get_from_module
def get(identifier):
    return get_from_module(identifier, globals(), 'objective')
