# -*- coding: utf-8 -*-
from __future__ import absolute_import

from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine import Layer
from keras.engine import InputSpec
from keras import backend as K


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
    def __init__(self, alpha_initializer=0.2,
                 beta_initializer=5.0,
                 alpha_regularizer=None,
                 alpha_constraint=None,
                 beta_regularizer=None,
                 beta_constraint=None,
                 shared_axes=None,
                 **kwargs):
        super(ParametricSoftplus, self).__init__(**kwargs)
        self.supports_masking = True
        self.alpha_initializer = initializers.get(alpha_initializer)
        self.alpha_regularizer = regularizers.get(alpha_regularizer)
        self.alpha_constraint = constraints.get(alpha_constraint)
        self.beta_initializer = initializers.get(beta_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        if shared_axes is None:
            self.shared_axes = None
        elif not isinstance(shared_axes, (list, tuple)):
            self.shared_axes = [shared_axes]
        else:
            self.shared_axes = list(shared_axes)

    def build(self, input_shape):
        param_shape = list(input_shape[1:])
        self.param_broadcast = [False] * len(param_shape)
        if self.shared_axes is not None:
            for i in self.shared_axes:
                param_shape[i - 1] = 1
                self.param_broadcast[i - 1] = True

        self.alpha = self.add_weight(param_shape,
                                     name='alpha',
                                     initializer=self.alpha_initializer,
                                     regularizer=self.alpha_regularizer,
                                     constraint=self.alpha_constraint)
        self.beta = self.add_weight(param_shape,
                                     name='beta',
                                     initializer=self.beta_initializer,
                                     regularizer=self.beta_regularizer,
                                     constraint=self.beta_constraint)

        # Set input spec
        axes = {}
        if self.shared_axes:
            for i in range(1, len(input_shape)):
                if i not in self.shared_axes:
                    axes[i] = input_shape[i]
        self.input_spec = InputSpec(ndim=len(input_shape), axes=axes)
        self.built = True

    def call(self, x, mask=None):
        if K.backend() == 'theano':
            return K.softplus(K.pattern_broadcast(self.beta, self.param_broadcast) * x) * K.pattern_broadcast(self.alpha, self.param_broadcast)
        else:
            return K.softplus(self.beta * x) * self.alpha

    def get_config(self):
        config = {
            'alpha_initializer': initializers.serialize(self.alpha_initializer),
            'alpha_regularizer': regularizers.serialize(self.alpha_regularizer),
            'alpha_constraint': constraints.serialize(self.alpha_constraint),
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'shared_axes': self.shared_axes
        }
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
    def __init__(self, init='one', power_init=1, weights=None, axis=-1, fit=True, **kwargs):
        self.supports_masking = True
        self.init = initializations.get(init)
        self.initial_weights = weights
        self.axis = axis
        self.power_init = power_init
        self.fit = fit
        super(PowerPReLU, self).__init__(**kwargs)

    def build(self, input_shape):
        alpha_shape = input_shape[self.axis]

        self.alpha_pos = self.init((alpha_shape,),
                                name='{}alpha_pos'.format(self.name))
        self.alpha_neg = self.init((alpha_shape,),
                                name='{}alpha_neg'.format(self.name))
        self.beta_pos = K.variable(np.zeros(alpha_shape),
                                 name='{}beta_pos'.format(self.name))
        self.beta_neg = K.variable(np.zeros(alpha_shape),
                                 name='{}beta_neg'.format(self.name))
        self.rho_pos = K.variable(self.power_init * np.ones(alpha_shape),
                                 name='{}rho_pos'.format(self.name))
        self.rho_neg = K.variable(self.power_init * np.ones(alpha_shape),
                                 name='{}rho_neg'.format(self.name))
        if self.fit:
            self.trainable_weights = [self.alpha_pos, self.alpha_neg, self.beta_pos, self.beta_neg, self.rho_pos, self.rho_neg]

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
        beta_pos = K.reshape(self.beta_pos, broadcast_shape)
        beta_neg = K.reshape(self.beta_neg, broadcast_shape)
        rho_pos = K.reshape(self.rho_pos, broadcast_shape)
        rho_neg = K.reshape(self.rho_neg, broadcast_shape)
        pos = alpha_pos * K.pow(K.relu(x + beta_pos) + K.epsilon(), rho_pos)
        neg = alpha_neg * K.pow(K.relu(-x + beta_neg) + K.epsilon(), rho_neg)
        return pos + neg

    def get_config(self):
        config = {'init': self.init.__name__}
        base_config = super(PowerPReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PowerReLU(Layer):
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
    def __init__(self, init='one', power_init=1, weights=None, axis=-1, fit=True, **kwargs):
        self.supports_masking = True
        self.init = initializations.get(init)
        self.initial_weights = weights
        self.axis = axis
        self.power_init = power_init
        self.fit = fit
        super(PowerReLU, self).__init__(**kwargs)

    def build(self, input_shape):
        alpha_shape = input_shape[self.axis]

        self.alpha = self.init((alpha_shape,),
                                name='{}alpha_pos'.format(self.name))
        self.rho = K.variable(self.power_init * np.ones(alpha_shape),
                                 name='{}rho_pos'.format(self.name))

        if self.fit:
            self.trainable_weights = [self.alpha, self.rho]

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
        alpha = K.reshape(self.alpha, broadcast_shape)
        rho = K.reshape(self.rho, broadcast_shape)

        return alpha * K.pow(K.relu(x) + K.epsilon(), rho)

    def get_config(self):
        config = {'init': self.init.__name__}
        base_config = super(PowerReLU, self).get_config()
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


class Hill(Layer):

    def __init__(self, a_initializer='ones',
                 k_initializer='ones',
                 n_initializer='ones',
                 z_initializer='zeros',
                 a_regularizer=None,
                 a_constraint=constraints.NonNeg(),
                 k_regularizer=None,
                 k_constraint=constraints.NonNeg(),
                 n_regularizer=None,
                 n_constraint=constraints.NonNeg(),  
                 z_regularizer=None,
                 z_constraint=constraints.NonNeg(),
                 shared_axes=None,
                 a_shared=True,
                 k_shared=True,
                 n_shared=True,
                 z_shared=True,
                 z_one=False,
                 **kwargs):
        super(Hill, self).__init__(**kwargs)
        self.supports_masking = True
        self.a_initializer = initializers.get(a_initializer)
        self.a_regularizer = regularizers.get(a_regularizer)
        self.a_constraint = constraints.get(a_constraint)
        self.k_initializer = initializers.get(a_initializer)
        self.k_regularizer = regularizers.get(a_regularizer)
        self.k_constraint = constraints.get(a_constraint)
        self.n_initializer = initializers.get(a_initializer)
        self.n_regularizer = regularizers.get(a_regularizer)
        self.n_constraint = constraints.get(a_constraint)
        self.z_initializer = initializers.get(a_initializer)
        self.z_regularizer = regularizers.get(a_regularizer)
        self.z_constraint = constraints.get(a_constraint)
        self.a_shared = a_shared
        self.k_shared = k_shared
        self.n_shared = n_shared
        self.z_shared = z_shared
        self.z_one = z_one
        if shared_axes is None:
            self.shared_axes = None
        elif not isinstance(shared_axes, (list, tuple)):
            self.shared_axes = [shared_axes]
        else:
            self.shared_axes = list(shared_axes)

    def build(self, input_shape):
        param_shape_unshared = list(input_shape[1:])
        param_shape_shared = list(input_shape[1:])
        param_broadcast_unshared = [False] * len(param_shape_unshared)
        param_broadcast_shared = [False] * len(param_shape_shared)
        if self.shared_axes is not None:
            for i in self.shared_axes:
                param_shape_shared[i - 1] = 1
                param_broadcast_shared[i - 1] = True

        self.a_param_broadcast = param_broadcast_shared
        self.k_param_broadcast = param_broadcast_shared
        self.n_param_broadcast = param_broadcast_shared
        self.z_param_broadcast = param_broadcast_shared
        a_param_shape = param_shape_shared
        k_param_shape = param_shape_shared
        n_param_shape = param_shape_shared
        z_param_shape = param_shape_shared

        if not self.a_shared:
            a_param_shape = param_shape_unshared
            self.a_param_broadcast = param_broadcast_unshared

        if not self.k_shared:
            k_param_shape = param_shape_unshared
            self.k_param_broadcast = param_broadcast_unshared

        if not self.n_shared:
            n_param_shape = param_shape_unshared
            self.n_param_broadcast = param_broadcast_unshared

        if not self.z_shared:
            z_param_shape = param_shape_unshared
            self.z_param_broadcast = param_broadcast_unshared

        self.a = self.add_weight(shape=a_param_shape,
                                     name='a',
                                     initializer=self.a_initializer,
                                     regularizer=self.a_regularizer,
                                     constraint=self.a_constraint)
        self.k = self.add_weight(shape=k_param_shape,
                                     name='k',
                                     initializer=self.k_initializer,
                                     regularizer=self.k_regularizer,
                                     constraint=self.k_constraint)
        self.n = self.add_weight(shape=n_param_shape,
                                     name='n',
                                     initializer=self.n_initializer,
                                     regularizer=self.n_regularizer,
                                     constraint=self.n_constraint)
        if not self.z_one:
            self.z = self.add_weight(shape=z_param_shape,
                                         name='z',
                                         initializer=self.z_initializer,
                                         regularizer=self.z_regularizer,
                                         constraint=self.z_constraint)
        else:
            self.z = K.constant(1.)
        # Set input spec
        axes = {}
        if self.shared_axes:
            for i in range(1, len(input_shape)):
                if i not in self.shared_axes:
                    axes[i] = input_shape[i]
        self.input_spec = InputSpec(ndim=len(input_shape), axes=axes)
        self.built = True

    def call(self, inputs, mask=None):
        if K.backend() == 'theano':
            a = K.pattern_broadcast(self.a, self.a_param_broadcast)
            k = K.pattern_broadcast(self.k, self.k_param_broadcast)
            n = K.pattern_broadcast(self.n, self.n_param_broadcast)
            z = K.pattern_broadcast(self.z, self.z_param_broadcast)
        else:
            a = self.a
            k = self.k
            n = self.n
            z = self.z
        return a / (K.pow((k / (inputs + 1e-5)), n) + z + 1e-5)

    def get_config(self):
        config = {
            'a_initializer': initializers.serialize(self.a_initializer),
            'a_regularizer': regularizers.serialize(self.a_regularizer),
            'a_constraint': constraints.serialize(self.a_constraint),
            'k_initializer': initializers.serialize(self.k_initializer),
            'k_regularizer': regularizers.serialize(self.k_regularizer),
            'k_constraint': constraints.serialize(self.k_constraint),
            'n_initializer': initializers.serialize(self.n_initializer),
            'n_regularizer': regularizers.serialize(self.n_regularizer),
            'n_constraint': constraints.serialize(self.n_constraint),
            'z_initializer': initializers.serialize(self.z_initializer),
            'z_regularizer': regularizers.serialize(self.z_regularizer),
            'z_constraint': constraints.serialize(self.z_constraint),
            'shared_axes': self.shared_axes
        }
        base_config = super(Hill, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
