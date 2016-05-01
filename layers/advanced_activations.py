from keras import initializations
from keras.layers.core import Layer
from keras import backend as K
import numpy as np

class ParametricSoftplus(Layer):
    '''Parametric Softplus of the form: alpha * log(1 + exp(beta * X))

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as the input.

    # Arguments
        alpha_init: float. Initial value of the alpha weights.
        beta_init: float. Initial values of the beta weights.
        weights: initial weights, as a list of 2 numpy arrays.

    # References:
        - [Inferring Nonlinear Neuronal Computation Based on Physiologically Plausible Inputs](http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003143)
    '''
    def __init__(self, alpha_init=0.2, beta_init=5.0,
                 weights=None, fit=True, **kwargs):
        self.supports_masking = True
        self.alpha_init = K.cast_to_floatx(alpha_init)
        self.beta_init = K.cast_to_floatx(beta_init)
        self.initial_weights = weights
        self.fit = fit
        super(ParametricSoftplus, self).__init__(**kwargs)

    def build(self, input_shape):
        input_shape = input_shape[1:]
        self.alphas = K.variable(self.alpha_init * np.ones(input_shape),
                                 name='{}_alphas'.format(self.name))
        self.betas = K.variable(self.beta_init * np.ones(input_shape),
                                name='{}_betas'.format(self.name))
        if self.fit:
            self.trainable_weights = [self.alphas, self.betas]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        return K.softplus(self.betas * x) * self.alphas

    def get_config(self):
        config = {'alpha_init': self.alpha_init,
                  'beta_init': self.beta_init}
        base_config = super(ParametricSoftplus, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))