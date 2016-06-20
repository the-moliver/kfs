from __future__ import absolute_import
from keras.engine import Layer
from keras import backend as K


class CoupledGaussianDropout(Layer):
    '''CoupledGaussianDropout    '''
    def __init__(self, **kwargs):
        self.supports_masking = True
        self.uses_learning_phase = True
        super(CoupledGaussianDropout, self).__init__(**kwargs)

    def call(self, x, mask=None):
        noise_x = x + K.random_normal(shape=K.shape(x),
                                      mean=0.0,
                                      std=K.sqrt(K.sqrt(K.square(x) + K.epsilon())))
        return K.in_train_phase(noise_x, x)

    def get_config(self):
        config = {}
        base_config = super(CoupledGaussianDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CoupledGaussianDropoutRect(Layer):
    '''CoupledGaussianDropout    '''
    def __init__(self, **kwargs):
        self.supports_masking = True
        self.uses_learning_phase = True
        super(CoupledGaussianDropoutRect, self).__init__(**kwargs)

    def call(self, x, mask=None):
        noise_x = K.relu(x + K.random_normal(shape=K.shape(x),
                                             mean=0.0,
                                             std=K.sqrt(K.sqrt(K.square(x) + K.epsilon()))))
        return K.in_train_phase(noise_x, x)

    def get_config(self):
        config = {}
        base_config = super(CoupledGaussianDropoutRect, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Dropout(Layer):
    '''Applies Dropout to the input. Dropout consists in randomly setting
    a fraction `p` of input units to 0 at each update during training time,
    which helps prevent overfitting.

    # Arguments
        p: float between 0 and 1. Fraction of the input units to drop.

    # References
        - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
    '''
    def __init__(self, p, exclude_axis=None, **kwargs):
        self.supports_masking = True
        self.uses_learning_phase = True
        self.p = p
        self.exclude_axis = exclude_axis
        super(Dropout, self).__init__(**kwargs)

    def call(self, x, mask=None):
        if 0. < self.p < 1.:
            x = K.in_train_phase(dropout(x, level=self.p, exclude_axis=self.exclude_axis), x)
        return x

    def get_config(self):
        config = {'p': self.p, 'exclude_axis': self.exclude_axis}
        base_config = super(Dropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class Gain(Layer):
    '''Multiplicative constant gain
    '''
    def __init__(self, gain=.01, **kwargs):
        self.gain = gain
        self.supports_masking = True
        self.uses_learning_phase = False
        super(Gain, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return self.gain * x

    def get_config(self):
        config = {'gain': self.gain}
        base_config = super(Gain, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


import theano
import numpy as np
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.shared_randomstreams import RandomStreams as RandomStreams2


def reshape(x, shape):
    return T.reshape(x, shape)


def dropout(x, level, exclude_axis=None, seed=None):
    if level < 0. or level >= 1:
        raise Exception('Dropout level must be in interval [0, 1].')
    if seed is None:
        seed = np.random.randint(10e6)
    rng = RandomStreams2(seed=seed)
    retain_prob = 1. - level
    if exclude_axis == 0:
        xshape = x[0].shape
        mask = T.zeros(xshape)
        mask = rng.shuffle_row_elements(T.set_subtensor(mask[:T.cast(retain_prob*xshape[0], 'int64')], 1))
        mask = reshape(mask, (1, -1))
        mask = T.addbroadcast(mask, 0)
    # if exclude_axis == 1:
    #     xshape = x[:, 0].shape
    #     mask = reshape(rng.binomial(xshape, p=retain_prob, dtype=x.dtype), (xshape[0], 1, -1))
    #     mask = T.addbroadcast(mask, 1)
    elif exclude_axis == [0, 1]:
        xshape = x[0, 0].shape
        # mask = reshape(rng.binomial(xshape, p=retain_prob, dtype=x.dtype), (1, 1, -1))
        mask = T.zeros(xshape)
        mask = rng.shuffle_row_elements(T.set_subtensor(mask[:T.cast(retain_prob*xshape[0], 'int64')], 1))
        mask = reshape(mask, (1, 1, -1))
        mask = T.addbroadcast(mask, 0, 1)
    else:
        mask = rng.binomial(x.shape, p=retain_prob, dtype=x.dtype)
    x *= mask
    x /= retain_prob
    return x
