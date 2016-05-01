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
