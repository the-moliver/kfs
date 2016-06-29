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
    def __init__(self, m):
        self.m = m

    def __call__(self, p):
        sumaxes = tuple(range(1, np.ndim(p)))
        v = p[:self.m]
        w = p[self.m:2*self.m]
        v2 = w - v*K.sum(v*w, axis=sumaxes, keepdims=True)/K.sum(v*v, axis=sumaxes, keepdims=True)

        norms = K.sqrt(K.sum(v**2 + v2**2, axis=sumaxes, keepdims=True))

        v /= norms
        v2 /= norms

        p = T.set_subtensor(p[:self.m], v)
        p = T.set_subtensor(p[self.m:2*self.m], v2)

        return p

    def get_config(self):
        return {'name': self.__class__.__name__,
                'm': self.m}




from keras.utils.generic_utils import get_from_module
def get(identifier, kwargs=None):
    return get_from_module(identifier, globals(), 'constraint',
                           instantiate=True, kwargs=kwargs)
