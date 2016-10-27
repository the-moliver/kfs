from __future__ import absolute_import
from keras.engine import Layer
from keras import backend as K
from keras.layers.core import Dropout


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


class ShapeDropout(Dropout):
    '''This version performs the same function as Dropout, however it drops
    entire 2D feature maps instead of individual elements. If adjacent pixels
    within feature maps are strongly correlated (as is normally the case in
    early convolution layers) then regular dropout will not regularize the
    activations and will otherwise just result in an effective learning rate
    decrease. In this case, SpatialDropout2D will help promote independence
    between feature maps and should be used instead.
    # Arguments
        p: float between 0 and 1. Fraction of the input units to drop.
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode is it at index 3.
            It defaults to the `image_dim_ordering` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "tf".
    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.
    # Output shape
        Same as input
    # References
        - [Efficient Object Localization Using Convolutional Networks](https://arxiv.org/pdf/1411.4280.pdf)
    '''
    def __init__(self, p, noise_shape=None, **kwargs):
        self.noise_shape = noise_shape
        super(ShapeDropout, self).__init__(p, **kwargs)

    def _get_noise_shape(self, x):
        return self.noise_shape



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