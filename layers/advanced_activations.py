from keras import initializations
from keras.layers.core import Layer, InputSpec
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

        if self.fit:
            self.alphas = K.variable(self.alpha_init * np.ones(input_shape),
                                 name='{}_alphas'.format(self.name))
            self.betas = K.variable(self.beta_init * np.ones(input_shape),
                                name='{}_betas'.format(self.name))
            self.trainable_weights = [self.alphas, self.betas]
        else:
            self.alphas = K.variable(self.alpha_init,
                                 name='{}_alphas'.format(self.name))
            self.betas = K.variable(self.beta_init,
                                name='{}_betas'.format(self.name))

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


class PowerPReLU(Layer):
    '''Parametric Rectified Linear Unit:
    `f(x) = alphas * x for x < 0`,
    `f(x) = x for x >= 0`,
    where `alphas` is a learned array with the same shape as x.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as the input.

    # Arguments
        init: initialization function for the weights.
        weights: initial weights, as a list of a single Numpy array.

    # References
        - [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](http://arxiv.org/pdf/1502.01852v1.pdf)
    '''
    def __init__(self, init='one', weights=None, axis=-1, **kwargs):
        self.supports_masking = True
        self.init = initializations.get(init)
        self.initial_weights = weights
        self.axis = axis
        super(PowerPReLU, self).__init__(**kwargs)

    def build(self, input_shape):
        alpha_shape = input_shape[self.axis]

        self.alpha_pos = self.init((alpha_shape,),
                                name='{}alpha_pos'.format(self.name))
        self.alpha_neg = self.init((alpha_shape,),
                                name='{}alpha_neg'.format(self.name))
        # self.rho_pos = self.init((alpha_shape,),
        #                         name='{}rho_pos'.format(self.name))
        # self.rho_neg = self.init((alpha_shape,),
        #                         name='{}rho_neg'.format(self.name))
        self.rho_pos = K.variable(2 * np.ones(alpha_shape),
                                 name='{}rho_pos'.format(self.name))
        self.rho_neg = K.variable(2 * np.ones(alpha_shape),
                                 name='{}rho_neg'.format(self.name))
        self.trainable_weights = [self.alpha_pos, self.alpha_neg, self.rho_pos, self.rho_neg]

        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=input_shape)]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]
        alpha_pos = K.reshape(self.alpha_pos, broadcast_shape)
        alpha_neg = K.reshape(self.alpha_neg, broadcast_shape)
        rho_pos = K.reshape(self.rho_pos, broadcast_shape)
        rho_neg = K.reshape(self.rho_neg, broadcast_shape)
        pos = alpha_pos * K.pow(K.relu(x) + K.epsilon(), rho_pos)
        neg = alpha_neg * K.pow(K.relu(-x) + K.epsilon(), rho_neg)
        return pos + neg

    def get_config(self):
        config = {'init': self.init.__name__}
        base_config = super(PowerPReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class Polynomial(Layer):
    '''Parametric Rectified Linear Unit:
    `f(x) = alphas * x for x < 0`,
    `f(x) = x for x >= 0`,
    where `alphas` is a learned array with the same shape as x.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as the input.

    # Arguments
        init: initialization function for the weights.
        weights: initial weights, as a list of a single Numpy array.

    # References
        - [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](http://arxiv.org/pdf/1502.01852v1.pdf)
    '''
    def __init__(self, degree=2, init='zero', init1='one', weights=None, **kwargs):
        self.supports_masking = True
        self.init1 = initializations.get(init1)
        self.init = initializations.get(init)
        self.initial_weights = weights
        self.degree = degree
        super(Polynomial, self).__init__(**kwargs)

    def build(self, input_shape):
        self.offset = self.init(input_shape[1:],
                                 name='{}offset'.format(self.name))
        self.alphas0 = self.init(input_shape[1:],
                                 name='{}_alphas_0'.format(self.name))
        self.alphas1 = self.init1(input_shape[1:],
                                  name='{}_alphas_1'.format(self.name))
        self.alphas2 = self.init(input_shape[1:],
                                 name='{}_alphas_2'.format(self.name))

        self.trainable_weights = [self.offset, self.alphas0, self.alphas1, self.alphas2]

        if self.degree > 2:
            self.alphas3 = self.init(input_shape[1:],
                                     name='{}_alphas_3'.format(self.name))
            self.trainable_weights.append(self.alphas3)

        if self.degree > 3:
            self.alphas4 = self.init(input_shape[1:],
                                     name='{}_alphas_4'.format(self.name))
            self.trainable_weights.append(self.alphas4)

        if self.degree > 4:
            self.alphas5 = self.init(input_shape[1:],
                                     name='{}_alphas_5'.format(self.name))
            self.trainable_weights.append(self.alphas5)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        x -= self.offset
        out = self.alphas0 + self.alphas1*x + self.alphas2*x*x
        if self.degree > 2:
            out += self.alphas3*x*x*x
        if self.degree > 3:
            out += self.alphas4*x*x*x*x
        if self.degree > 4:
            out += self.alphas5*x*x*x*x*x

        return out

    def get_config(self):
        config = {'init': self.init.__name__}
        base_config = super(Polynomial, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
