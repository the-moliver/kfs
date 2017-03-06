from __future__ import absolute_import
from keras import backend as K
from keras.constraints import Constraint
import numpy as np
import six
if K.backend() == 'theano':
    from theano import tensor as T


class UnitNormOrthogonal(Constraint):
    '''Constrain pairs of weights incident to be orthogonal and to have unit norm.

    # Arguments
        m: Int; The number of pairs to be made orthogonal. Anything beyond the
                specified pairs be made unit norm, but is assumed unpaired
        dim_ordering: {'th' or 'tf'} determines whether the pairs on on the
                      first or last axes
    '''
    def __init__(self, m, dim_ordering=None):
        self.m = m
        self.dim_ordering = dim_ordering

    def __call__(self, p):
        if self.dim_ordering == 'tf':
            rotate_axis = range(K.ndim(p))
            rotate_axis = [rotate_axis[-1]] + rotate_axis[:-1]
            p = K.permute_dimensions(p, rotate_axis)

        sumaxes = tuple(range(1, K.ndim(p)))

        v = p[:self.m]
        w = p[self.m:2*self.m]
        x = p[2*self.m:]

        v2 = w - v*K.sum(v*w, axis=sumaxes, keepdims=True)/K.sum(v*v, axis=sumaxes, keepdims=True)

        norms_paired = K.sqrt(K.sum(v**2 + v2**2, axis=sumaxes, keepdims=True))
        norms_single = K.sqrt(K.sum(x**2, axis=sumaxes, keepdims=True))

        v /= norms_paired
        v2 /= norms_paired
        x /= norms_single

        out = K.concatenate((v, v2, x), axis=0)

        if self.dim_ordering == 'tf':
            rotate_axis = range(K.ndim(out))
            rotate_axis = rotate_axis[1:] + [rotate_axis[0]]
            out = K.permute_dimensions(out, rotate_axis)
        return out

    def get_config(self):
        return {'name': self.__class__.__name__,
                'm': self.m}


class Stochastic(Constraint):
    '''Constrain the weights incident to be positive and sum to one.

    # Arguments
        axis: integer, axis along which to calculate sums.
    '''
    def __init__(self, axis=0):
        self.axis = axis

    def __call__(self, p):
        p *= K.cast(p >= 0., K.floatx())
        return p / (K.epsilon() + K.sum(p, axis=self.axis, keepdims=True))

    def get_config(self):
        return {'name': self.__class__.__name__,
                'axis': self.axis}


def get(identifier):
    if identifier is None:
        return None
    if isinstance(identifier, dict):
        return deserialize(identifier)
    elif isinstance(identifier, six.string_types):
        config = {'class_name': str(identifier), 'config': {}}
        return deserialize(config)
    elif callable(identifier):
        return identifier
    else:
        raise ValueError('Could not interpret constraint identifier:',
                         identifier)
