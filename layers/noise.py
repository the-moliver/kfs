from __future__ import absolute_import
from keras.engine import Layer
from keras import backend as K
from keras.layers.core import Dropout


class CoupledGaussianDropout(Layer):
    """Apply Gaussian noise where variance is equal activation.

    As it is a regularization layer, it is only active at training time.

    # Arguments
        rate: float, drop probability (as with `Dropout`).
            The multiplicative noise will have
            standard deviation `sqrt(rate / (1 - rate))`.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as input.

    # References
        - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting Srivastava, Hinton, et al. 2014](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
    """

    def __init__(self, factor=1, **kwargs):
        super(CoupledGaussianDropout, self).__init__(**kwargs)
        self.supports_masking = True
        self.factor = factor

    def call(self, inputs, training=None):
        def noised():
            return inputs + K.random_normal(shape=K.shape(inputs),
                                            mean=0.0,
                                            stddev=self.factor * K.sqrt(K.abs(inputs) + K.epsilon()))
        return K.in_train_phase(noised, inputs, training=training)

    def get_config(self):
        config = {'rate': self.rate}
        base_config = super(CoupledGaussianDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AxesDropout(Dropout):
    '''This version performs the same function as Dropout, however it drops
    entire axes instead of individual elements. If adjacent pixels
    within feature maps are strongly correlated (as is normally the case in
    early convolution layers) then regular dropout will not regularize the
    activations and will otherwise just result in an effective learning rate
    decrease. In this case, AxesDropout will help promote independence
    between feature maps and should be used instead.
    # Arguments
        rate: float between 0 and 1. Fraction of the input units to drop.

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.
    # Output shape
        Same as input

    '''
    def __init__(self, rate, axes=None, **kwargs):
        super(AxesDropout, self).__init__(rate, **kwargs)
        self.axes = axes

    def _get_noise_shape(self, inputs):
        input_shape = K.shape(inputs)
        noise_shape = [1]*K.ndim(inputs)
        for i in self.axes:
            noise_shape[i] = input_shape[i]
        return noise_shape


class Gain(Layer):
    '''Multiplicative constant gain
    '''
    def __init__(self, gain=.01, **kwargs):
        super(Gain, self).__init__(**kwargs)
        self.gain = gain
        self.supports_masking = True

    def call(self, inputs, training=None):
        return self.gain * inputs

    def get_config(self):
        config = {'gain': self.gain}
        base_config = super(Gain, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
