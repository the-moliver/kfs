# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division

import numpy as np

from collections import OrderedDict
import copy
from six.moves import zip

from keras import backend as K
from keras import activations, initializations, regularizers, constraints
from keras.regularizers import ActivityRegularizer
from keras.engine import InputSpec, Layer, Merge

import marshal
import types
import sys


class GQM(Layer):
    def __init__(self, quadratic_filters=2, init='glorot_uniform', weights=None,
                 W_quad_regularizer=None, W_lin_regularizer=None, activity_regularizer=None,
                 W_quad_constraint=None, W_lin_constraint=None,
                 bias=True, input_dim=None, **kwargs):
        self.init = initializations.get(init)
        self.quadratic_filters = quadratic_filters
        self.input_dim = input_dim

        self.W_quad_regularizer = regularizers.get(W_quad_regularizer)
        self.W_lin_regularizer = regularizers.get(W_lin_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_quad_constraint = constraints.get(W_quad_constraint)
        self.W_lin_constraint = constraints.get(W_lin_constraint)

        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=2)]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(GQM, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, input_dim))]

        self.W_quad = self.init((input_dim, self.quadratic_filters),
                                name='{}_W_quad'.format(self.name))
        self.W_lin = self.init((input_dim, 1),
                               name='{}_W_lin'.format(self.name))

        self.trainable_weights = [self.W_quad, self.W_lin]

        self.regularizers = []
        if self.W_quad_regularizer:
            self.W_quad_regularizer.set_param(self.W_quad)
            self.regularizers.append(self.W_quad_regularizer)

        if self.W_lin_regularizer:
            self.W_lin_regularizer.set_param(self.W_lin)
            self.regularizers.append(self.W_lin_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        self.constraints = {}
        if self.W_quad_constraint:
            self.constraints[self.W_quad] = self.W_quad_constraint
        if self.W_lin_constraint:
            self.constraints[self.W_lin] = self.W_lin_constraint

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        output_quad = K.square(K.dot(x, self.W_quad))
        output_lin = K.dot(x, self.W_lin)

        return K.concatenate([output_quad, output_lin], axis=1)

    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], self.quadratic_filters + 1)

    def get_config(self):
        config = {'quadratic_filters': self.quadratic_filters,
                  'init': self.init.__name__,
                  'W_quad_regularizer': self.W_quad_regularizer.get_config() if self.W_quad_regularizer else None,
                  'W_lin_regularizer': self.W_lin_regularizer.get_config() if self.W_lin_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_quad_constraint': self.W_quad_constraint.get_config() if self.W_quad_constraint else None,
                  'W_lin_constraint': self.W_lin_constraint.get_config() if self.W_lin_constraint else None,
                  'input_dim': self.input_dim}
        base_config = super(GQM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RustSTC(Layer):
    def __init__(self, quadratic_filters_ex=2, quadratic_filters_sup=2, init='glorot_uniform', weights=None,
                 W_quad_ex_regularizer=None, W_quad_sup_regularizer=None, W_lin_regularizer=None, activity_regularizer=None,
                 W_quad_ex_constraint=None, W_quad_sup_constraint=None, W_lin_constraint=None,
                 bias=True, input_dim=None, **kwargs):
        self.init = initializations.get(init)
        self.quadratic_filters_ex = quadratic_filters_ex
        self.quadratic_filters_sup = quadratic_filters_sup
        self.input_dim = input_dim

        self.W_quad_ex_regularizer = regularizers.get(W_quad_ex_regularizer)
        self.W_quad_sup_regularizer = regularizers.get(W_quad_sup_regularizer)
        self.W_lin_regularizer = regularizers.get(W_lin_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_quad_ex_constraint = constraints.get(W_quad_ex_constraint)
        self.W_quad_sup_constraint = constraints.get(W_quad_sup_constraint)
        self.W_lin_constraint = constraints.get(W_lin_constraint)

        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=2)]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(RustSTC, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, input_dim))]

        self.W_quad_ex = self.init((input_dim, self.quadratic_filters_ex),
                                name='{}_W_quad_ex'.format(self.name))
        self.W_quad_sup = self.init((input_dim, self.quadratic_filters_sup),
                                name='{}_W_quad_sup'.format(self.name))
        self.W_lin = self.init((input_dim, 1),
                               name='{}_W_lin'.format(self.name))

        self.trainable_weights = [self.W_quad_ex, self.W_quad_sup, self.W_lin]

        self.regularizers = []
        if self.W_quad_ex_regularizer:
            self.W_quad_ex_regularizer.set_param(self.W_quad_ex)
            self.regularizers.append(self.W_quad_ex_regularizer)

        if self.W_quad_sup_regularizer:
            self.W_quad_sup_regularizer.set_param(self.W_quad_sup)
            self.regularizers.append(self.W_quad_sup_regularizer)

        if self.W_lin_regularizer:
            self.W_lin_regularizer.set_param(self.W_lin)
            self.regularizers.append(self.W_lin_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        self.constraints = {}
        if self.W_quad_ex_constraint:
            self.constraints[self.W_quad_ex] = self.W_quad_ex_constraint
        if self.W_quad_sup_constraint:
            self.constraints[self.W_quad_sup] = self.W_quad_sup_constraint
        if self.W_lin_constraint:
            self.constraints[self.W_lin] = self.W_lin_constraint

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        output_quad_ex = K.square(K.dot(x, self.W_quad_ex))
        output_quad_sup = K.square(K.dot(x, self.W_quad_sup))
        output_lin = K.square(K.relu(K.dot(x, self.W_lin)))

        ex = K.sqrt(output_lin + K.sum(output_quad_ex, axis=1, keepdims=True))
        sup = K.sqrt(K.sum(output_quad_sup, axis=1, keepdims=True))
        return K.concatenate([ex, sup], axis=1)

    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], 2)

    def get_config(self):
        config = {'quadratic_filters': self.quadratic_filters,
                  'init': self.init.__name__,
                  'W_quad_ex_regularizer': self.W_quad_ex_regularizer.get_config() if self.W_quad_ex_regularizer else None,
                  'W_quad_sup_regularizer': self.W_quad_sup_regularizer.get_config() if self.W_quad_sup_regularizer else None,
                  'W_lin_regularizer': self.W_lin_regularizer.get_config() if self.W_lin_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_quad_ex_constraint': self.W_quad_ex_constraint.get_config() if self.W_quad_ex_constraint else None,
                  'W_quad_sup_constraint': self.W_quad_sup_constraint.get_config() if self.W_quad_sup_constraint else None,
                  'W_lin_constraint': self.W_lin_constraint.get_config() if self.W_lin_constraint else None,
                  'input_dim': self.input_dim}
        base_config = super(RustSTC, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class NakaRushton(Layer):
    def __init__(self, weights=None, input_dim=None, **kwargs):
        # self.init = initializations.get(init)
        self.input_dim = input_dim

        # self.W_quad_regularizer = regularizers.get(W_quad_regularizer)
        # self.W_lin_regularizer = regularizers.get(W_lin_regularizer)
        # self.activity_regularizer = regularizers.get(activity_regularizer)

        # self.W_quad_constraint = constraints.get(W_quad_constraint)
        # self.W_lin_constraint = constraints.get(W_lin_constraint)

        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=2)]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(NakaRushton, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        assert input_shape[1] == 2
        input_dim = input_shape[1]
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, input_dim))]

        # self.W = self.init((6,),
        #                         name='{}_W'.format(self.name))
        self.alpha = K.variable(0.,
                                name='{}_alpha'.format(self.name))
        self.beta_delta = K.variable(np.array([[1.], [-1.]]),
                                name='{}_beta_delta'.format(self.name))
        self.gamma_eta = K.variable(np.array([[.01], [-.01]]),
                                name='{}_gamma_eta'.format(self.name))
        self.rho = K.variable(1.,
                                name='{}_rho'.format(self.name))

        self.trainable_weights = [self.alpha, self.beta_delta, self.gamma_eta, self.rho]


        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        x = K.pow(K.relu(x) + K.epsilon(), self.rho)
        output = self.alpha + (K.dot(x, self.beta_delta) / (K.dot(x, self.gamma_eta) + 1.))
        return output

    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], 1)

    def get_config(self):
        config = {'init': self.init.__name__,
                  'input_dim': self.input_dim}
        base_config = super(NakaRushton, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class EminusS(Layer):
    def __init__(self, activation='linear', weights=None,
                 b_regularizer=None, activity_regularizer=None,
                 b_constraint=None,
                 bias=True, input_dim=None, **kwargs):

        self.activation = activations.get(activation)
        self.input_dim = input_dim

        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=2)]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(EminusS, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        assert input_shape[1] == 2
        input_dim = input_shape[1]
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, input_dim))]

        self.W = K.variable(np.array([[1], [-1]]),
                            name='{}_W'.format(self.name))
        if self.bias:
            self.b = K.zeros((1,),
                             name='{}_b'.format(self.name))
            self.trainable_weights = [self.b]

        self.regularizers = []

        if self.bias and self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        self.constraints = {}
        if self.bias and self.b_constraint:
            self.constraints[self.b] = self.b_constraint

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        output = K.dot(x, self.W)
        if self.bias:
            output += self.b
        return self.activation(output)

    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], 1)

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'bias': self.bias,
                  'input_dim': self.input_dim}
        base_config = super(EminusS, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GQM_conv(Layer):
    def __init__(self, quadratic_filters=2, init='glorot_uniform', weights=None,
                 W_quad_regularizer=None, W_lin_regularizer=None, activity_regularizer=None,
                 W_quad_constraint=None, W_lin_constraint=None,
                 bias=True, input_dim=None, **kwargs):
        self.init = initializations.get(init)
        self.quadratic_filters = quadratic_filters
        self.input_dim = input_dim

        self.W_quad_regularizer = regularizers.get(W_quad_regularizer)
        self.W_lin_regularizer = regularizers.get(W_lin_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_quad_constraint = constraints.get(W_quad_constraint)
        self.W_lin_constraint = constraints.get(W_lin_constraint)

        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=5)]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(GQM_conv, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 5
        input_dim = input_shape[1]*input_shape[2]
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, input_shape[1], input_shape[2], input_shape[3], input_shape[4]))]

        self.W_quad = self.init((input_dim, self.quadratic_filters),
                                name='{}_W_quad'.format(self.name))
        self.W_lin = self.init((input_dim, 1),
                               name='{}_W_lin'.format(self.name))

        self.trainable_weights = [self.W_quad, self.W_lin]

        self.regularizers = []
        if self.W_quad_regularizer:
            self.W_quad_regularizer.set_param(self.W_quad)
            self.regularizers.append(self.W_quad_regularizer)

        if self.W_lin_regularizer:
            self.W_lin_regularizer.set_param(self.W_lin)
            self.regularizers.append(self.W_lin_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        self.constraints = {}
        if self.W_quad_constraint:
            self.constraints[self.W_quad] = self.W_quad_constraint
        if self.W_lin_constraint:
            self.constraints[self.W_lin] = self.W_lin_constraint

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        x = K.permute_dimensions(x, [0, 3, 4, 1, 2])  # (samples, time, feat, X, Y)
        x = K.reshape(x, (-1, self.input_spec[0].shape[1]*self.input_spec[0].shape[2]))  # (samples * X * Y, time * feat)
        output_quad = K.square(K.dot(x, self.W_quad))
        output_lin = K.dot(x, self.W_lin)
        output = K.permute_dimensions(K.reshape(K.concatenate([output_quad, output_lin], axis=1), (-1, 1, self.input_spec[0].shape[3], self.input_spec[0].shape[4], self.quadratic_filters+1)), (0, 1, 4, 2, 3))
        return output

    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) == 5
        return (input_shape[0], 1, self.quadratic_filters + 1, self.input_spec[0].shape[3], self.input_spec[0].shape[4])

    def get_config(self):
        config = {'quadratic_filters': self.quadratic_filters,
                  'init': self.init.__name__,
                  'W_quad_regularizer': self.W_quad_regularizer.get_config() if self.W_quad_regularizer else None,
                  'W_lin_regularizer': self.W_lin_regularizer.get_config() if self.W_lin_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_quad_constraint': self.W_quad_constraint.get_config() if self.W_quad_constraint else None,
                  'W_lin_constraint': self.W_lin_constraint.get_config() if self.W_lin_constraint else None,
                  'input_dim': self.input_dim}
        base_config = super(GQM_conv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GQM_4D(Layer):
    def __init__(self, quadratic_filters=2, init='glorot_uniform', weights=None,
                 W_quad_regularizer=None, W_lin_regularizer=None, activity_regularizer=None,
                 W_quad_constraint=None, W_lin_constraint=None,
                 bias=True, input_dim=None, **kwargs):
        self.init = initializations.get(init)
        self.quadratic_filters = quadratic_filters
        self.input_dim = input_dim

        self.W_quad_regularizer = regularizers.get(W_quad_regularizer)
        self.W_lin_regularizer = regularizers.get(W_lin_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_quad_constraint = constraints.get(W_quad_constraint)
        self.W_lin_constraint = constraints.get(W_lin_constraint)

        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=5)]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(GQM_4D, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 5
        input_dim1 = input_shape[1]
        input_dim2 = input_shape[2]
        input_dim3 = input_shape[3]
        input_dim4 = input_shape[4]
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, input_dim1, input_dim2, input_dim3, input_dim4))]

        self.W_quad = self.init((input_dim1, input_dim2, input_dim3, input_dim4, self.quadratic_filters),
                                name='{}_W_quad'.format(self.name))
        self.W_lin = self.init((input_dim1, input_dim2, input_dim3, input_dim4, 1),
                               name='{}_W_lin'.format(self.name))

        self.trainable_weights = [self.W_quad, self.W_lin]

        self.regularizers = []
        if self.W_quad_regularizer:
            self.W_quad_regularizer.set_param(self.W_quad)
            self.regularizers.append(self.W_quad_regularizer)

        if self.W_lin_regularizer:
            self.W_lin_regularizer.set_param(self.W_lin)
            self.regularizers.append(self.W_lin_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        self.constraints = {}
        if self.W_quad_constraint:
            self.constraints[self.W_quad] = self.W_quad_constraint
        if self.W_lin_constraint:
            self.constraints[self.W_lin] = self.W_lin_constraint

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        xshape = x.shape
        x = K.reshape(x, (-1, xshape[1]*xshape[2]*xshape[3]*xshape[4]))
        W_quad = K.reshape(self.W_quad, (-1, self.quadratic_filters))
        W_lin = K.reshape(self.W_lin, (-1, 1))
        output_quad = K.square(K.dot(x, W_quad))
        output_lin = K.dot(x, W_lin)

        return K.concatenate([output_quad, output_lin], axis=1)

    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) == 5
        return (input_shape[0], self.quadratic_filters + 1)

    def get_config(self):
        config = {'quadratic_filters': self.quadratic_filters,
                  'init': self.init.__name__,
                  'W_quad_regularizer': self.W_quad_regularizer.get_config() if self.W_quad_regularizer else None,
                  'W_lin_regularizer': self.W_lin_regularizer.get_config() if self.W_lin_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_quad_constraint': self.W_quad_constraint.get_config() if self.W_quad_constraint else None,
                  'W_lin_constraint': self.W_lin_constraint.get_config() if self.W_lin_constraint else None,
                  'input_dim': self.input_dim}
        base_config = super(GQM_4D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
