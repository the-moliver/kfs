from __future__ import absolute_import
from keras import backend as K
import numpy as np
import six
from keras.utils.generic_utils import serialize_keras_object
from keras.utils.generic_utils import deserialize_keras_object
from keras.constraints import *
# if K.backend() == 'theano':
#     from theano import tensor as T


class UnitNormOrthogonal(Constraint):
    '''Constrain pairs of weights incident to be orthogonal and to have unit norm.

    # Arguments
        m: Int; The number of pairs to be made orthogonal. Anything beyond the
                specified pairs be made unit norm, but is assumed unpaired

    '''
    def __init__(self, m, singles=False, interleave=False):
        self.m = m
        self.singles = singles
        self.interleave = interleave
        # self.data_format = data_format

    def __call__(self, p):
        # if self.data_format == 'channels_last':
        rotate_axis = list(range(K.ndim(p)))
        rotate_axis = [rotate_axis[-1]] + rotate_axis[:-1]
        p = K.permute_dimensions(p, rotate_axis)

        sumaxes = tuple(range(1, K.ndim(p)))

        if self.interleave:
            if self.singles:
                v = p[::3]
                w = p[1::3]
            else:
                v = p[::2]
                w = p[1::2]
        else:
            v = p[:self.m]
            w = p[self.m:2*self.m]

        v2 = w - v*K.sum(v*w, axis=sumaxes, keepdims=True)/K.sum(v*v, axis=sumaxes, keepdims=True)

        norms_paired = K.sqrt(K.sum(v**2 + v2**2, axis=sumaxes, keepdims=True))
        v /= norms_paired
        v2 /= norms_paired

        if self.singles:
            if self.interleave:
                x = p[2::3]
            else:
                x = p[2*self.m:]
            norms_single = K.sqrt(K.sum(x**2, axis=sumaxes, keepdims=True))
            x /= norms_single
            out = K.concatenate((v, v2, x), axis=0)
        else:
            out = K.concatenate((v, v2), axis=0)

        # if self.dim_ordering == 'tf':
        rotate_axis = list(range(K.ndim(out)))
        rotate_axis = rotate_axis[1:] + [rotate_axis[0]]
        out = K.permute_dimensions(out, rotate_axis)
        return out

    def get_config(self):
        return {'m': self.m,
                'singles': self.singles,
                'interleave': self.interleave}


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


class Symmetric(Constraint):
    '''Constrain the weight tensor to be symmetric in last 2 dimensions'''
    def __call__(self, w):
        axes = range(K.ndim(w))
        axes = axes[:-2]+axes[-1:]+axes[-2:-1]
        w = .5*(w + K.permute_dimensions(w, axes))
        return w
