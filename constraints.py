from __future__ import absolute_import
from keras import backend as K
from keras.constraints import Constraint
import theano
import numpy as np
from theano import tensor as T


class UnitNormOrthogonal(Constraint):
    '''Constrain the weights incident to each hidden unit to have unit norm.

    # Arguments
        axis: integer, axis along which to calculate weight norms. For instance,
            in a `Dense` layer the weight matrix has shape (input_dim, output_dim),
            set `axis` to `0` to constrain each weight vector of length (input_dim).
            In a `MaxoutDense` layer the weight tensor has shape (nb_feature, input_dim, output_dim),
            set `axis` to `1` to constrain each weight vector of length (input_dim),
            i.e. constrain the filters incident to the `max` operation.
            In a `Convolution2D` layer with the Theano backend, the weight tensor
            has shape (nb_filter, stack_size, nb_row, nb_col), set `axis` to `[1,2,3]`
            to constrain the weights of each filter tensor of size (stack_size, nb_row, nb_col).
            In a `Convolution2D` layer with the TensorFlow backend, the weight tensor
            has shape (nb_row, nb_col, stack_size, nb_filter), set `axis` to `[0,1,2]`
            to constrain the weights of each filter tensor of size (nb_row, nb_col, stack_size).
    '''
    def __init__(self, m, axis=0):
        self.m = m
        self.axis = axis

    def __call__(self, p):
        if self.axis == 0:
            v = p[:self.m]
            w = p[self.m:]
        elif self.axis == 1:
            v = p[:, :self.m]
            w = p[:, self.m:]
        v2 = w - v*K.sum(v*w, axis=self.axis, keepdims=True)/K.sum(v*v, axis=self.axis, keepdims=True)
        if self.axis == 0:
            p = T.set_subtensor(p[self.m:], v2)
        elif self.axis == 1:
            p = T.set_subtensor(p[:, self.m:], v2)

        return p / K.sqrt(K.sum(p**2, axis=self.axis, keepdims=True))

    def get_config(self):
        return {'name': self.__class__.__name__,
                'm': self.m,
                'axis': self.axis}




from keras.utils.generic_utils import get_from_module
def get(identifier, kwargs=None):
    return get_from_module(identifier, globals(), 'constraint',
                           instantiate=True, kwargs=kwargs)
