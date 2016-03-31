from __future__ import absolute_import
from keras.layers.core import MaskedLayer
from keras import backend as K


class CoupledGaussianDropout(MaskedLayer):
    '''CoupledGaussianDropout    '''
    def __init__(self, **kwargs):
        super(CoupledGaussianDropout, self).__init__(**kwargs)

    def get_output(self, train):
        X = self.get_input(train)
        if train:
            X += K.random_normal(shape=K.shape(X), mean=0.0,
                                 std=K.sqrt(K.sqrt(K.square(X) + K.epsilon())))
        return X

    def get_config(self):
        config = {"name": self.__class__.__name__}
        base_config = super(CoupledGaussianDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CoupledGaussianDropoutRect(MaskedLayer):
    '''CoupledGaussianDropout    '''
    def __init__(self, **kwargs):
        super(CoupledGaussianDropoutRect, self).__init__(**kwargs)

    def get_output(self, train):
        X = self.get_input(train)
        if train:
            X = K.relu(X + K.random_normal(shape=K.shape(X), mean=0.0,
                                 std=K.sqrt(K.sqrt(K.square(X) + K.epsilon()))))
        return X

    def get_config(self):
        config = {"name": self.__class__.__name__}
        base_config = super(CoupledGaussianDropoutRect, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
