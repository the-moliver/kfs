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


# class TemporalFilter(Layer):
#     '''Apply a Dense layer to each dimension[1] (time_dimension) input.
#     Useful for learning a lower dimensional set of temporal filters'.

#     # Input shape
#         3D tensor with shape `(nb_sample, time_dimension, input_dim)`.

#     # Output shape
#         3D tensor with shape `(nb_sample, time_dimension, output_dim)`.

#     # Arguments
#         output_dim: int > 0.
#         init: name of initialization function for the weights of the layer
#             (see [initializations](../initializations.md)),
#             or alternatively, Theano function to use for weights
#             initialization. This parameter is only relevant
#             if you don't pass a `weights` argument.
#         activation: name of activation function to use
#             (see [activations](../activations.md)),
#             or alternatively, elementwise Theano function.
#             If you don't specify anything, no activation is applied
#             (ie. "linear" activation: a(x) = x).
#         weights: list of numpy arrays to set as initial weights.
#             The list should have 2 elements, of shape `(input_dim, output_dim)`
#             and (output_dim,) for weights and biases respectively.
#         W_regularizer: instance of [WeightRegularizer](../regularizers.md)
#             (eg. L1 or L2 regularization), applied to the main weights matrix.
#         b_regularizer: instance of [WeightRegularizer](../regularizers.md),
#             applied to the bias.
#         activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
#             applied to the network output.
#         W_constraint: instance of the [constraints](../constraints.md) module
#             (eg. maxnorm, nonneg), applied to the main weights matrix.
#         b_constraint: instance of the [constraints](../constraints.md) module,
#             applied to the bias.
#         input_dim: dimensionality of the input (integer).
#             This argument (or alternatively, the keyword argument `input_shape`)
#             is required when using this layer as the first layer in a model.
#     '''
#     input_ndim = 3

#     def __init__(self, output_dim,
#                  init='glorot_uniform', activation='linear', weights=None,
#                  W_regularizer=None, b_regularizer=None, activity_regularizer=None,
#                  W_constraint=None, b_constraint=None,
#                  input_dim=None, input_length=None, **kwargs):
#         self.output_dim = output_dim
#         self.init = initializations.get(init)
#         self.activation = activations.get(activation)

#         self.W_regularizer = regularizers.get(W_regularizer)
#         self.b_regularizer = regularizers.get(b_regularizer)
#         self.activity_regularizer = regularizers.get(activity_regularizer)

#         self.W_constraint = constraints.get(W_constraint)
#         self.b_constraint = constraints.get(b_constraint)
#         self.constraints = [self.W_constraint, self.b_constraint]

#         self.initial_weights = weights

#         self.input_dim = input_dim
#         self.input_length = input_length
#         if self.input_dim:
#             kwargs['input_shape'] = (self.input_length, self.input_dim)
#         super(TemporalFilter, self).__init__(**kwargs)

#     def build(self):
#         input_dim = self.input_shape[1]

#         self.W = self.init((input_dim, self.output_dim),
#                            name='{}_W'.format(self.name))
#         self.b = K.zeros((self.output_dim,),
#                          name='{}_b'.format(self.name))

#         self.trainable_weights = [self.W, self.b]
#         self.regularizers = []

#         if self.W_regularizer:
#             self.W_regularizer.set_param(self.W)
#             self.regularizers.append(self.W_regularizer)

#         if self.b_regularizer:
#             self.b_regularizer.set_param(self.b)
#             self.regularizers.append(self.b_regularizer)

#         if self.activity_regularizer:
#             self.activity_regularizer.set_layer(self)
#             self.regularizers.append(self.activity_regularizer)

#         if self.initial_weights is not None:
#             self.set_weights(self.initial_weights)
#             del self.initial_weights

#     @property
#     def output_shape(self):
#         input_shape = self.input_shape
#         return (input_shape[0], self.output_dim, input_shape[2])

#     def get_output(self, train=False):
#         X = self.get_input(train)  # (samples, timesteps, input_dim)
#         # Squash samples and timesteps into a single axis
#         x = K.permute_dimensions(X, [0, 2, 1])
#         x = K.reshape(x, (-1, self.input_shape[-2]))  # (samples * input_dim, timesteps)
#         Y = K.dot(x, self.W) + self.b  # (samples * input_dim, output_dim)
#         Y = K.reshape(Y, (-1, self.input_shape[-1], self.output_dim))  # (samples, input_dim, output_dim)
#         Y = K.permute_dimensions(Y, [0, 2, 1])  # (samples, output_dim, input_dim)
#         Y = self.activation(Y)
#         return Y

#     def get_config(self):
#         config = {'name': self.__class__.__name__,
#                   'output_dim': self.output_dim,
#                   'init': self.init.__name__,
#                   'activation': self.activation.__name__,
#                   'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
#                   'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
#                   'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
#                   'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
#                   'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
#                   'input_dim': self.input_dim,
#                   'input_length': self.input_length}
#         base_config = super(TemporalFilter, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))


class TemporalFilter(Layer):
    '''Apply a Dense layer to each dimension[1] (time_dimension) input.
    Useful for learning a lower dimensional set of temporal filters'.

    # Input shape
        3D tensor with shape `(nb_sample, time_dimension, input_dim)`.

    # Output shape
        3D tensor with shape `(nb_sample, time_dimension, output_dim)`.

    # Arguments
        output_dim: int > 0.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)),
            or alternatively, Theano function to use for weights
            initialization. This parameter is only relevant
            if you don't pass a `weights` argument.
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list of numpy arrays to set as initial weights.
            The list should have 2 elements, of shape `(input_dim, output_dim)`
            and (output_dim,) for weights and biases respectively.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        b_constraint: instance of the [constraints](../constraints.md) module,
            applied to the bias.
        input_dim: dimensionality of the input (integer).
            This argument (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.
    '''
    def __init__(self, output_dim, init='glorot_uniform', activation='linear', weights=None,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, input_dim=None, **kwargs):
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=3)]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(TemporalFilter, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        input_dim = input_shape[1]
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, input_dim, input_shape[2]))]
        self.W = self.init((input_dim, self.output_dim),
                           name='{}_W'.format(self.name))
        if self.bias:
            self.b = K.zeros((self.output_dim,),
                             name='{}_b'.format(self.name))
            self.trainable_weights = [self.W, self.b]
        else:
            self.trainable_weights = [self.W]

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.bias and self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        self.constraints = {}
        if self.W_constraint:
            self.constraints[self.W] = self.W_constraint
        if self.bias and self.b_constraint:
            self.constraints[self.b] = self.b_constraint

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        # Squash samples and timesteps into a single axis
        x = K.permute_dimensions(x, [0, 2, 1])
        x = K.reshape(x, (-1, self.input_spec[0].shape[-2]))  # (samples * input_dim, timesteps)
        output = K.dot(x, self.W)  # (samples * input_dim, output_dim)
        if self.bias:
            output += self.b
        output = K.reshape(output, (-1, self.input_spec[0].shape[-1], self.output_dim))  # (samples, input_dim, output_dim)
        output = K.permute_dimensions(output, [0, 2, 1])  # (samples, output_dim, input_dim)
        return self.activation(output)

    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) == 3
        return (input_shape[0], self.output_dim, input_shape[2])

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'bias': self.bias,
                  'input_dim': self.input_dim}
        base_config = super(TemporalFilter, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class SoftMinMax(Layer):
    '''Computes a selective and adaptive soft-min or soft-max over the inputs.

    # Input shape
        2D tensor with shape: `(nb_samples, input_dim)`.

    # Output shape
        2D tensor with shape: `(nb_samples, output_dim)`.

    # Arguments
        output_dim: int > 0.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)),
            or alternatively, Theano function to use for weights
            initialization. This parameter is only relevant
            if you don't pass a `weights` argument.
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list of numpy arrays to set as initial weights.
            The list should have 1 element, of shape `(input_dim, output_dim)`.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        b_constraint: instance of the [constraints](../constraints.md) module,
            applied to the bias.
        input_dim: dimensionality of the input (integer).
            This argument (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.
    '''
    input_ndim = 2

    def __init__(self, output_dim, init='glorot_uniform', activation='linear', weights=None,
                 W_regularizer=None, k_regularizer=None, activity_regularizer=None,
                 W_constraint=None, k_constraint=None, input_dim=None, **kwargs):
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.output_dim = output_dim

        self.W_regularizer = regularizers.get(W_regularizer)
        self.k_regularizer = regularizers.get(k_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.k_constraint = constraints.get(k_constraint)
        self.constraints = [self.W_constraint, self.k_constraint]

        self.initial_weights = weights

        self.input_dim = input_dim
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        self.input = K.placeholder(ndim=2)
        super(SoftMinMax, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[1]

        self.W = self.init((input_dim, self.output_dim),
                           name='{}_W'.format(self.name))
        self.k = K.zeros((self.output_dim,),
                         name='{}_k'.format(self.name))

        self.trainable_weights = [self.W, self.k]

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.k_regularizer:
            self.k_regularizer.set_param(self.k)
            self.regularizers.append(self.k_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    @property
    def output_shape(self):
        return (self.input_shape[0], self.output_dim)

    def get_output(self, train=False):
        X = self.get_input(train)
        W = K.softplus(10.*self.W)/10.
        kX = self.k[None, None, :] * X[:, :, None]
        kX = K.clip(kX, -30, 30)
        wekx = W[None, :, :] * K.exp(kX)
        output = ((X[:, :, None] * wekx).sum(axis=1) / (wekx.sum(axis=1) + 1))

        return output

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'k_regularizer': self.k_regularizer.get_config() if self.k_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'k_constraint': self.k_constraint.get_config() if self.k_constraint else None,
                  'input_dim': self.input_dim}
        base_config = super(SoftMinMax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))




class DenseNonNeg(Layer):
    '''Just your regular fully connected NN layer.

    # Input shape
        2D tensor with shape: `(nb_samples, input_dim)`.

    # Output shape
        2D tensor with shape: `(nb_samples, output_dim)`.

    # Arguments
        output_dim: int > 0.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)),
            or alternatively, Theano function to use for weights
            initialization. This parameter is only relevant
            if you don't pass a `weights` argument.
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list of numpy arrays to set as initial weights.
            The list should have 2 elements, of shape `(input_dim, output_dim)`
            and (output_dim,) for weights and biases respectively.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        b_constraint: instance of the [constraints](../constraints.md) module,
            applied to the bias.
        input_dim: dimensionality of the input (integer).
            This argument (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.
    '''
    input_ndim = 2

    def __init__(self, output_dim, init='glorot_uniform', activation='linear', weights=None,
                 W_regularizer=None, activity_regularizer=None,
                 W_constraint=None, input_dim=None, **kwargs):
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.output_dim = output_dim

        self.W_regularizer = regularizers.get(W_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.constraints = [self.W_constraint]

        self.initial_weights = weights

        self.input_dim = input_dim
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(DenseNonNeg, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[1]

        self.W = self.init((input_dim, self.output_dim),
                           name='{}_W'.format(self.name))

        self.trainable_weights = [self.W]

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    @property
    def output_shape(self):
        return (self.input_shape[0], self.output_dim)

    def get_output(self, train=False):
        X = self.get_input(train)
        W = K.softplus(10.*self.W)/10.
        output = self.activation(K.dot(X, W))
        return output

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'input_dim': self.input_dim}
        base_config = super(DenseNonNeg, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class DenseEnergy(Layer):
    '''Just your regular fully connected NN layer.

    # Input shape
        2D tensor with shape: `(nb_samples, input_dim)`.

    # Output shape
        2D tensor with shape: `(nb_samples, output_dim)`.

    # Arguments
        output_dim: int > 0.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)),
            or alternatively, Theano function to use for weights
            initialization. This parameter is only relevant
            if you don't pass a `weights` argument.
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list of numpy arrays to set as initial weights.
            The list should have 1 element, of shape `(input_dim, output_dim)`.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        b_constraint: instance of the [constraints](../constraints.md) module,
            applied to the bias.
        input_dim: dimensionality of the input (integer).
            This argument (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.
    '''
    input_ndim = 2

    def __init__(self, output_dim, init='glorot_uniform', activation='linear', weights=None,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None, input_dim=None, **kwargs):
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.output_dim = output_dim

        self.W1_regularizer = regularizers.get(W_regularizer)
        self.W2_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.constraints = [self.W_constraint, self.W_constraint, self.b_constraint]

        self.initial_weights = weights

        self.input_dim = input_dim
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        self.input = K.placeholder(ndim=2)
        super(DenseEnergy, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[1]

        self.W1 = self.init((input_dim, self.output_dim))
        self.W2 = self.init((input_dim, self.output_dim))
        self.b = K.zeros((self.output_dim,))

        self.params = [self.W1, self.W2, self.b]

        self.regularizers = []
        if self.W1_regularizer:
            self.W1_regularizer.set_param(self.W1)
            self.regularizers.append(self.W1_regularizer)
            self.W2_regularizer.set_param(self.W2)
            self.regularizers.append(self.W2_regularizer)

        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    @property
    def output_shape(self):
        return (self.input_shape[0], self.output_dim)

    def get_output(self, train=False):
        X = self.get_input(train)
        # MW = self.W1 + self.W2
        # W1 = self.W1 - MW*K.sum(MW*self.W1, axis=0, keepdims=True)/K.sum(MW*MW, axis=0, keepdims=True)
        # W2 = self.W2 - W1*K.sum(W1*self.W2, axis=0, keepdims=True)/K.sum(W1*W1, axis=0, keepdims=True)
        if train:
            W1 = self.W1
            W2 = self.W2 - W1*K.sum(W1*self.W2, axis=0, keepdims=True)/K.sum(W1*W1, axis=0, keepdims=True)
            norm = K.sqrt(K.sum(W1**2 + W2**2, axis=0, keepdims=True))
            self.W1 = W1 / norm
            self.W2 = W2 / norm
            W1 = self.W1
            W2 = self.W2
        else:
            W1 = self.W1
            W2 = self.W2

        output = K.sqrt(K.square(K.dot(X, W1)) + K.square(K.dot(X, W2)) + K.epsilon()) + self.b
        return output

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'W1_regularizer': self.W1_regularizer.get_config() if self.W1_regularizer else None,
                  'W2_regularizer': self.W2_regularizer.get_config() if self.W2_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'input_dim': self.input_dim}
        base_config = super(DenseEnergy, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Convolution2DEnergy(Layer):
    '''Convolution operator for filtering windows of two-dimensional inputs.
    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(3, 128, 128)` for 128x128 RGB pictures.

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.

    # Output shape
        4D tensor with shape:
        `(samples, nb_filter, nb_row, nb_col)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, nb_row, nb_col, nb_filter)` if dim_ordering='tf'.


    # Arguments
        nb_filter: Number of convolution filters to use.
        nb_row: Number of rows in the convolution kernel.
        nb_col: Number of columns in the convolution kernel.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)), or alternatively,
            Theano function to use for weights initialization.
            This parameter is only relevant if you don't pass
            a `weights` argument.
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list of numpy arrays to set as initial weights.
        border_mode: 'valid' or 'same'.
        subsample: tuple of length 2. Factor by which to subsample output.
            Also called strides elsewhere.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        b_constraint: instance of the [constraints](../constraints.md) module,
            applied to the bias.
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode is it at index 3.
    '''
    input_ndim = 4

    def __init__(self, nb_filter, nb_row, nb_col,
                 init='glorot_uniform', activation='linear', weights=None,
                 border_mode='valid', subsample=(1, 1), dim_ordering='th',
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None, **kwargs):

        if border_mode not in {'valid', 'same'}:
            raise Exception('Invalid border mode for Convolution2D:', border_mode)
        self.nb_filter = nb_filter
        self.nb_row = nb_row
        self.nb_col = nb_col
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        assert border_mode in {'valid', 'same'}, 'border_mode must be in {valid, same}'
        self.border_mode = border_mode
        self.subsample = tuple(subsample)
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering

        self.W1_regularizer = regularizers.get(W_regularizer)
        self.W2_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.constraints = [self.W_constraint, self.W_constraint, self.b_constraint]

        self.initial_weights = weights
        self.input = K.placeholder(ndim=4)
        super(Convolution2DEnergy, self).__init__(**kwargs)

    def build(self):
        if self.dim_ordering == 'th':
            stack_size = self.input_shape[1]
            self.W_shape = (self.nb_filter, stack_size, self.nb_row, self.nb_col)
        elif self.dim_ordering == 'tf':
            stack_size = self.input_shape[3]
            self.W_shape = (self.nb_row, self.nb_col, stack_size, self.nb_filter)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        self.W1 = self.init(self.W_shape)
        self.W2 = self.init(self.W_shape)
        self.b = K.zeros((self.nb_filter,))
        self.params = [self.W1, self.W2, self.b]
        self.regularizers = []

        if self.W1_regularizer:
            self.W1_regularizer.set_param(self.W1)
            self.regularizers.append(self.W_regularizer)
            self.W2_regularizer.set_param(self.W2)
            self.regularizers.append(self.W2_regularizer)

        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    @property
    def output_shape(self):
        input_shape = self.input_shape
        if self.dim_ordering == 'th':
            rows = input_shape[2]
            cols = input_shape[3]
        elif self.dim_ordering == 'tf':
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        rows = conv_output_length(rows, self.nb_row,
                                  self.border_mode, self.subsample[0])
        cols = conv_output_length(cols, self.nb_col,
                                  self.border_mode, self.subsample[1])

        if self.dim_ordering == 'th':
            return (input_shape[0], self.nb_filter, rows, cols)
        elif self.dim_ordering == 'tf':
            return (input_shape[0], rows, cols, self.nb_filter)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def get_output(self, train=False):
        X = self.get_input(train)
        X = K.reshape(X, (self.input_shape[0]*self.input_shape[1], 1, self.input_shape[2], self.input_shape[3]))
        if self.dim_ordering == 'th':
            axis = [1, 2, 3]
        elif self.dim_ordering == 'tf':
        	axis = [0, 1, 2]
        W1 = self.W1
        W2 = self.W2 - W1*K.sum(W1*self.W2, axis=axis, keepdims=True)/K.sum(W1*W1, axis=axis, keepdims=True)
        conv_out1 = K.conv2d(X, W1, strides=self.subsample,
                            border_mode=self.border_mode,
                            dim_ordering=self.dim_ordering,
                            image_shape=self.input_shape,
                            filter_shape=self.W_shape)
        conv_out2 = K.conv2d(X, W2, strides=self.subsample,
                            border_mode=self.border_mode,
                            dim_ordering=self.dim_ordering,
                            image_shape=self.input_shape,
                            filter_shape=self.W_shape)
        conv_out = K.sqrt(K.square(conv_out1) + K.square(conv_out2) + K.epsilon())
        if self.dim_ordering == 'th':
            output = conv_out + K.reshape(self.b, (1, self.nb_filter, 1, 1))
        elif self.dim_ordering == 'tf':
            output = conv_out + K.reshape(self.b, (1, 1, 1, self.nb_filter))
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        output = K.reshape(output, (self.input_shape[0], self.input_shape[1]*output.shape[1], output.shape[2], output.shape[3]))
        return output

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'nb_filter': self.nb_filter,
                  'nb_row': self.nb_row,
                  'nb_col': self.nb_col,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'border_mode': self.border_mode,
                  'subsample': self.subsample,
                  'dim_ordering': self.dim_ordering,
                  'W1_regularizer': self.W1_regularizer.get_config() if self.W1_regularizer else None,
                  'W2_regularizer': self.W2_regularizer.get_config() if self.W2_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None}
        base_config = super(Convolution2DEnergy, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



def conv_output_length(input_length, filter_size, border_mode, stride):
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - filter_size + 1
    return (output_length + stride - 1) // stride



class Feedback(Layer):
    '''Just your regular fully connected NN layer.

    # Input shape
        2D tensor with shape: `(nb_samples, input_dim)`.

    # Output shape
        2D tensor with shape: `(nb_samples, output_dim)`.

    # Arguments
        output_dim: int > 0.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)),
            or alternatively, Theano function to use for weights
            initialization. This parameter is only relevant
            if you don't pass a `weights` argument.
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list of numpy arrays to set as initial weights.
            The list should have 1 element, of shape `(input_dim, output_dim)`.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        b_constraint: instance of the [constraints](../constraints.md) module,
            applied to the bias.
        input_dim: dimensionality of the input (integer).
            This argument (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.
    '''
    input_ndim = 2

    def __init__(self, init='orthogonal', activation='linear', weights=None,
                 W_regularizer=None, activity_regularizer=None,
                 W_constraint=None, input_dim=None, **kwargs):
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        

        self.W_regularizer = regularizers.get(W_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.constraints = [self.W_constraint]

        self.initial_weights = weights

        self.input_dim = input_dim
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        self.input = K.placeholder(ndim=2)
        super(Feedback, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[1]
        self.output_dim = input_dim
        self.W = self.init((input_dim, input_dim))

        self.params = [self.W]

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)


        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    @property
    def output_shape(self):
        return (self.input_shape[0], self.output_dim)

    def get_output(self, train=False):
        X = self.get_input(train)
        output = self.activation(X + K.dot(X, self.W))
        return output

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'input_dim': self.input_dim}
        base_config = super(Feedback, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DivisiveNormalization(Layer):
    '''Just your regular fully connected NN layer.

    # Input shape
        2D tensor with shape: `(nb_samples, input_dim)`.

    # Output shape
        2D tensor with shape: `(nb_samples, output_dim)`.

    # Arguments
        output_dim: int > 0.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)),
            or alternatively, Theano function to use for weights
            initialization. This parameter is only relevant
            if you don't pass a `weights` argument.
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list of numpy arrays to set as initial weights.
            The list should have 1 element, of shape `(input_dim, output_dim)`.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        b_constraint: instance of the [constraints](../constraints.md) module,
            applied to the bias.
        input_dim: dimensionality of the input (integer).
            This argument (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.
    '''
    input_ndim = 2

    def __init__(self, init='glorot_uniform', activation='linear', weights=None,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None, input_dim=None, **kwargs):
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.constraints = [self.W_constraint, self.b_constraint]

        self.initial_weights = weights

        self.input_dim = input_dim
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        self.input = K.placeholder(ndim=2)
        
        super(DivisiveNormalization, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[1]
        self.output_dim = input_dim
        self.W = self.init((input_dim, input_dim))
        self.b = K.zeros((self.output_dim,))

        self.params = [self.W, self.b]

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    @property
    def output_shape(self):
        return (self.input_shape[0], self.output_dim)

    def get_output(self, train=False):
        X = self.get_input(train)
        output = self.activation(X / (1. + K.dot(X, K.softplus(10*self.W)/10) + self.b))
        return output

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'input_dim': self.input_dim}
        base_config = super(DivisiveNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Triangular(Layer):
    '''Just your regular fully connected NN layer.

    # Input shape
        2D tensor with shape: `(nb_samples, input_dim)`.

    # Output shape
        2D tensor with shape: `(nb_samples, output_dim)`.

    # Arguments
        output_dim: int > 0.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)),
            or alternatively, Theano function to use for weights
            initialization. This parameter is only relevant
            if you don't pass a `weights` argument.
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list of numpy arrays to set as initial weights.
            The list should have 1 element, of shape `(input_dim, output_dim)`.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        b_constraint: instance of the [constraints](../constraints.md) module,
            applied to the bias.
        input_dim: dimensionality of the input (integer).
            This argument (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.
    '''
    input_ndim = 2

    def __init__(self, output_dim, d=1, init='glorot_uniform', weights=None,
                 W_regularizer=None, activity_regularizer=None, W_constraint=None, 
                 input_dim=None, **kwargs):

        self.init = initializations.get(init)
        self.output_dim = output_dim
        self.d = d

        self.W_regularizer = regularizers.get(W_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.constraints = [self.W_constraint]

        self.initial_weights = weights

        self.input_dim = input_dim
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        self.input = K.placeholder(ndim=2)
        super(Triangular, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[1]

        self.W = self.init((input_dim, self.output_dim))

        self.params = [self.W]

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    @property
    def output_shape(self):
        return (self.input_shape[0], self.output_dim)

    def get_output(self, train=False):
        X = self.get_input(train)
        output = K.sqrt(K.sum(K.square(X[:, :, None] - self.W[None, :, :]), axis=1))
        if self.d is not 1:
            output = K.pow(output, self.d)

        return -output

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'input_dim': self.input_dim}
        base_config = super(Triangular, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GaussianMixtureDensity(Layer):
    '''A layer for creating a Gaussian Mixture Density Network.
    It should only be used as the last layer of the network and in
    combination with GaussianMixtureDensityLoss

    # Example

    ```python
        # as first layer in a sequential model:
        model = Sequential()
        model.add(Dense(32, input_dim=16))
        # now the model will take as input arrays of shape (*, 16)
        # and output arrays of shape (*, 32)

        # this is equivalent to the above:
        model = Sequential()
        model.add(Dense(32, input_shape=(16,)))

        # after the first layer, you don't need to specify
        # the size of the input anymore:
        model.add(Dense(32))
    ```

    # Arguments
        output_dim: int > 0.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)),
            or alternatively, Theano function to use for weights
            initialization. This parameter is only relevant
            if you don't pass a `weights` argument.
        weights: list of numpy arrays to set as initial weights.
            The list should have 2 elements, of shape `(input_dim, output_dim)`
            and (output_dim,) for weights and biases respectively.
        bias: whether to include a bias (i.e. make the layer affine rather than linear).
        input_dim: dimensionality of the input (integer).
            This argument (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.

    # Input shape
        2D tensor with shape: `(nb_samples, input_dim)`.

    # Output shape
        2D tensor with shape: `(nb_samples, output_dim)`.
    '''
    def __init__(self, output_dim, num_components, init='glorot_uniform', weights=None,
                 bias=True, input_dim=None, **kwargs):
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.num_components = num_components

        self.bias = bias
        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=2)]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(Dense, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, input_dim))]

        self.W_mu = self.init((input_dim, num_components*self.output_dim),
                           name='{}_W_mu'.format(self.name))
        self.W_sigma = self.init((input_dim, num_components),
                           name='{}_W_sigma'.format(self.name))
        self.W_pi = self.init((input_dim, num_components),
                           name='{}_W_pi'.format(self.name))
        if self.bias:
            self.b_mu = K.zeros((self.num_components*self.output_dim,),
                             name='{}_b_mu'.format(self.name))
            self.b_sigma = K.zeros((self.num_components,),
                             name='{}_b_sigma'.format(self.name))
            self.b_pi = K.zeros((self.num_components,),
                             name='{}_b_pi'.format(self.name))
            self.trainable_weights = [self.W_mu, self.b_sigma, self.W_pi, self.b_mu, self.b_sigma, self.b_pi]
        else:
            self.trainable_weights = [self.W_mu, self.b_sigma, self.W_pi]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        output_mu = K.dot(x, self.W_mu)
        output_sigma = K.dot(x, self.W_sigma)
        output_pi = K.dot(x, self.W_pi)
        if self.bias:
            output_mu += self.b_mu
            output_sigma += self.b_sigma
            output_pi += self.b_pi
        return K.concatenate([output_mu, K.exp(output_sigma), K.softmax(output_pi)], axis=-1)

    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'bias': self.bias,
                  'input_dim': self.input_dim}
        base_config = super(Dense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

