# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division

import numpy as np

import copy
import inspect
import types as python_types
import warnings

from keras import backend as K
from keras import activations 
from keras import initializers
from keras import regularizers
from kfs import constraints as kconstraints
from keras import constraints
from keras.engine import InputSpec
from keras.engine import Layer
from keras.utils.generic_utils import func_dump
from keras.utils.generic_utils import func_load
from keras.utils.generic_utils import deserialize_keras_object
from keras.legacy import interfaces


class FilterDims(Layer):
    '''The layer lets you filter any arbitrary set of axes by projection onto a new axis.
    This is can be useful for reducing dimensionality and/or regularizing spatio-temporal models or other
    models of structured data.

    # Example

    ```python
        # As a temporal filter in a 5D spatio-temporal model with input shape (#samples, 12, 3, 30, 30)
        # The input has 12 time steps, 3 color channels and X and Y of size 30:

        model = Sequential()
        model.add(TimeDistributed(Convolution2D(10, 5, 5, activation='linear', subsample=(2, 2)), input_shape=(12, 3, 30, 30)))

        # The output from the previous layer has shape (#samples, 12, 10, 13, 13)
        # We can use FilterDims to filter the 12 time steps on axis 1 by projeciton onto a new axis of 5 dimensions with a 12x5 matrix:

        model.add(FilterDims(filters=5, sum_axes=[1], filter_axes=[1], bias=False))

        # The weights learned by FilterDims are a set of temporal filters on the output of the spatial convolutions
        # The output dimensionality is (#samples, 5, 10, 13, 13)
        # We can then use FilterDims to filter the 5 temporal dimensions and 10 convolutional filter feature map
        # dimensions to create 2 spatio-temporal filters with a 5x10x2 weight tensor:

        model.add(FilterDims(filters=2, sum_axes=[1, 2], filter_axes=[1, 2], bias=False))

        # The output dimensionality is (#samples, 2, 13, 13)
        # We can then use FilterDims to spatially filter each spatio-temporal dimension with a 2x13x13 tensor:

        model.add(FilterDims(filters=1, sum_axes=[2, 3], filter_axes=[1, 2, 3], bias=False))

        # We only sum over the last two spatial axes resutling in an output dimensionality of (#samples, 2)
    ```

    # Arguments
        filters: number of filters to apply.
        filter_axes: a list of the axes of the input to filter
        sum_axes: a list of the axes of the input that should be summed across after filtering
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
        weights: list of Numpy arrays to set as initial weights.
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
        bias: whether to include a bias (i.e. make the layer affine rather than linear).
        input_dim: dimensionality of the input (integer).
            This argument (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.

    # Input shape
        ND tensor with arbitrary shape.

    # Output shape
        ND tensor with shape determined by input and arguments.
    '''
    def __init__(self, filters,
                 sum_axes,
                 filter_axes,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_activation=None,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(FilterDims, self).__init__(**kwargs)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.activation = activations.get(activation)
        self.kernel_activation = activations.get(kernel_activation)
        self.filters = filters
        self.sum_axes = list(sum_axes)
        self.sum_axes.sort()
        self.filter_axes = list(filter_axes)
        self.filter_axes.sort()
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.use_bias = use_bias
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        ndim = len(input_shape)
        assert ndim >= 2

        kernel_shape = [1] * (ndim - 1)
        kernel_broadcast = [False] * (ndim - 1)
        bias_broadcast = [True] * (ndim - 1)
        for i in self.filter_axes:
            kernel_shape[i-1] = input_shape[i]

        if self.filters > 1:
            kernel_shape.append(self.filters)
            kernel_broadcast.append(False)
            bias_broadcast.append(False)

        for i in set(range(1, ndim)) - set(self.filter_axes):
            kernel_broadcast[i-1] = True

        kernel_shape = tuple(kernel_shape)
        self.kernel = self.add_weight(kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        if self.use_bias:
            bias_shape = [1] * (ndim - 1)
            for i in set(self.filter_axes) - set(self.sum_axes):
                bias_shape[i-1] = input_shape[i]
                bias_broadcast[i-1] = False

            if self.filters > 1:
                bias_shape.append(self.filters)

            bias_shape = tuple(bias_shape)
            self.bias = self.add_weight(bias_shape,
                                        initializer=self.kernel_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.kernel_broadcast = kernel_broadcast
        self.bias_broadcast = bias_broadcast
        self.built = True

    def call(self, x, mask=None):
        ndim = K.ndim(x)
        xshape = K.shape(x)
        W = self.kernel_activation(self.kernel)

        if self.filter_axes == self.sum_axes:
            ax1 = [a-1 for a in self.sum_axes]
            if self.filters > 1:
                ax1 = ax1 + list(set(range(ndim)) - set(ax1))
            else:
                ax1 = ax1 + list(set(range(ndim - 1)) - set(ax1))
            ax2 = list(set(range(ndim)) - set(self.sum_axes))
            permute_dims = list(range(len(ax2)))
            permute_dims.insert(self.sum_axes[0], len(ax2))
            outdims = [-1] + [xshape[a] for a in ax2[1:]] + [self.filters]
            ax2 = ax2 + self.sum_axes
            W = K.permute_dimensions(W, ax1)
            W = K.reshape(W, (-1, self.filters))
            x = K.permute_dimensions(x, ax2)
            x = K.reshape(x, (-1, K.shape(W)[0]))
            output = K.reshape(K.dot(x, W), outdims)
            if self.use_bias:
                b_broadcast = [i for j, i in enumerate(self.bias_broadcast) if j not in self.sum_axes]
                b = K.squeeze(self.bias, self.sum_axes[0])
                if len(self.sum_axes) > 1:
                    b = K.squeeze(b, self.sum_axes[1] - 1)
                if len(self.sum_axes) > 2:
                    b = K.squeeze(b, self.sum_axes[2] - 2)
                if K.backend() == 'theano':
                    output += K.pattern_broadcast(b, b_broadcast)
                else:
                    output += b
            output = K.permute_dimensions(output, permute_dims)

        elif self.filters > 1:
            # bcast = list(np.where(self.broadcast)[0])
            permute_dims = list(range(ndim + 1))
            permute_dims[self.sum_axes[0]] = ndim
            permute_dims[ndim] = self.sum_axes[0]

            if K.backend() == 'theano':
                output = K.sum(x[..., None] * K.pattern_broadcast(W, self.kernel_broadcast), axis=self.sum_axes, keepdims=True)
            else:
                output = K.sum(x[..., None] * W, axis=self.sum_axes, keepdims=True)

            if self.use_bias:
                if K.backend() == 'theano':
                    output += K.pattern_broadcast(self.bias, self.bias_broadcast)
                else:
                    output += self.bias
            output = K.squeeze(K.permute_dimensions(output, permute_dims), ndim)
            if len(self.sum_axes) > 1:
                output = K.squeeze(output, self.sum_axes[1])

        else:
            if K.backend() == 'theano':
                output = K.sum(x * K.pattern_broadcast(W, self.kernel_broadcast), axis=self.sum_axes, keepdims=True)
            else:
                output = K.sum(x * W, axis=self.sum_axes, keepdims=True)
            if self.use_bias:
                if K.backend() == 'theano':
                    output += K.pattern_broadcast(self.bias, self.bias_broadcast)
                else:
                    output += self.bias
            output = K.squeeze(output, self.sum_axes[0])
            if len(self.sum_axes) > 1:
                output = K.squeeze(output, self.sum_axes[1]-1)

        return self.activation(output)

    def compute_output_shape(self, input_shape):
        if self.filters > 1:
            ndim = len(input_shape)
            output_shape = [input_shape[0]] + [1] * (ndim-1)
            for i in set(range(1, ndim)) - set(self.sum_axes):
                output_shape[i] = input_shape[i]

            output_shape.append(self.filters)
            permute_dims = list(range(ndim + 1))
            permute_dims[self.sum_axes[0]] = ndim
            permute_dims[ndim] = self.sum_axes[0]
            output_shape = [output_shape[i] for i in permute_dims]
            output_shape.pop(ndim)
            if len(self.sum_axes) > 1:
                output_shape.pop(self.sum_axes[1])
        else:
            output_shape = input_shape
            output_shape = [output_shape[i] for i in set(range(len(input_shape))) - set(self.sum_axes)]
            if len(output_shape) == 1:
                output_shape.append(1)

        return tuple(output_shape)

    def get_config(self):
        config = {
            'filters': self.filters,
            'sum_axes': self.sum_axes,
            'filter_axes': self.filter_axes,
            'activation': activations.serialize(self.activation),
            'kernel_activation': activations.serialize(self.kernel_activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(FilterDims, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class FilterDimsV1(Layer):
    '''The layer lets you filter any arbitrary set of axes by projection onto a new axis.
    This is can be useful for reducing dimensionality and/or regularizing spatio-temporal models or other
    models of structured data.

    # Example

    ```python
        # As a temporal filter in a 5D spatio-temporal model with input shape (#samples, 12, 3, 30, 30)
        # The input has 12 time steps, 3 color channels and X and Y of size 30:

        model = Sequential()
        model.add(TimeDistributed(Convolution2D(10, 5, 5, activation='linear', subsample=(2, 2)), input_shape=(12, 3, 30, 30)))

        # The output from the previous layer has shape (#samples, 12, 10, 13, 13)
        # We can use FilterDims to filter the 12 time steps on axis 1 by projeciton onto a new axis of 5 dimensions with a 12x5 matrix:

        model.add(FilterDims(filters=5, sum_axes=[1], filter_axes=[1], bias=False))

        # The weights learned by FilterDims are a set of temporal filters on the output of the spatial convolutions
        # The output dimensionality is (#samples, 5, 10, 13, 13)
        # We can then use FilterDims to filter the 5 temporal dimensions and 10 convolutional filter feature map
        # dimensions to create 2 spatio-temporal filters with a 5x10x2 weight tensor:

        model.add(FilterDims(filters=2, sum_axes=[1, 2], filter_axes=[1, 2], bias=False))

        # The output dimensionality is (#samples, 2, 13, 13)
        # We can then use FilterDims to spatially filter each spatio-temporal dimension with a 2x13x13 tensor:

        model.add(FilterDims(filters=1, sum_axes=[2, 3], filter_axes=[1, 2, 3], bias=False))

        # We only sum over the last two spatial axes resutling in an output dimensionality of (#samples, 2)
    ```

    # Arguments
        filters: number of filters to apply.
        filter_axes: a list of the axes of the input to filter
        sum_axes: a list of the axes of the input that should be summed across after filtering
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
        weights: list of Numpy arrays to set as initial weights.
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
        bias: whether to include a bias (i.e. make the layer affine rather than linear).
        input_dim: dimensionality of the input (integer).
            This argument (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.

    # Input shape
        ND tensor with arbitrary shape.

    # Output shape
        ND tensor with shape determined by input and arguments.
    '''
    def __init__(self, filters_simple,
                 filters_complex,
                 sum_axes,
                 filter_axes,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_activation=None,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(FilterDimsV1, self).__init__(**kwargs)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.activation = activations.get(activation)
        self.kernel_activation = activations.get(kernel_activation)
        self.filters_simple = filters_simple
        self.filters_complex = filters_complex
        self.sum_axes = list(sum_axes)
        self.sum_axes.sort()
        self.filter_axes = list(filter_axes)
        self.filter_axes.sort()
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = kconstraints.UnitNormOrthogonal(self.filters_complex)
        self.bias_constraint = constraints.get(bias_constraint)
        self.use_bias = use_bias
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        ndim = len(input_shape)
        assert ndim >= 2

        kernel_shape = [1] * (ndim - 1)
        kernel_broadcast = [False] * (ndim - 1)
        bias_broadcast = [True] * (ndim - 1)
        for i in self.filter_axes:
            kernel_shape[i-1] = input_shape[i]

        kernel_shape.append(2 * self.filters_complex + self.filters_simple)
        kernel_broadcast.append(False)
        bias_broadcast.append(False)

        for i in set(range(1, ndim)) - set(self.filter_axes):
            kernel_broadcast[i-1] = True

        kernel_shape = tuple(kernel_shape)
        self.kernel = self.add_weight(kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        if self.use_bias:
            bias_shape = [1] * (ndim - 1)
            for i in set(self.filter_axes) - set(self.sum_axes):
                bias_shape[i-1] = input_shape[i]
                bias_broadcast[i-1] = False

            bias_shape.append(self.filters_complex + self.filters_simple)

            bias_shape = tuple(bias_shape)
            self.bias = self.add_weight(bias_shape,
                                        initializer=self.kernel_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.kernel_broadcast = kernel_broadcast
        self.bias_broadcast = bias_broadcast
        self.built = True

    def call(self, x, mask=None):
        ndim = K.ndim(x)
        xshape = K.shape(x)
        W = self.kernel_activation(self.kernel)

        if self.filter_axes == self.sum_axes:
            ax1 = [a-1 for a in self.sum_axes]
            ax1 = ax1 + list(set(range(ndim)) - set(ax1))
            ax2 = list(set(range(ndim)) - set(self.sum_axes))
            permute_dims = list(range(len(ax2)))
            permute_dims.insert(self.sum_axes[0], len(ax2))
            outdims = [-1] + [xshape[a] for a in ax2[1:]] + [self.filters_complex + self.filters_simple]
            ax2 = ax2 + self.sum_axes
            W = K.permute_dimensions(W, ax1)
            W = K.reshape(W, (-1, 2 * self.filters_complex + self.filters_simple))
            x = K.permute_dimensions(x, ax2)
            x = K.reshape(x, (-1, K.shape(W)[0]))
            output = K.dot(x, W)
            output_complex = K.sqrt(K.square(output[:, :self.filters_complex]) + K.square(output[:, self.filters_complex:2*self.filters_complex]) + K.epsilon())
            output_simple = output[:, 2*self.filters_complex:]
            output = K.reshape(K.concatenate([output_complex, output_simple], axis=1), outdims)
            if self.use_bias:
                b_broadcast = [i for j, i in enumerate(self.bias_broadcast) if j not in self.sum_axes]
                b = K.squeeze(self.bias, self.sum_axes[0])
                if len(self.sum_axes) > 1:
                    b = K.squeeze(b, self.sum_axes[1] - 1)
                if len(self.sum_axes) > 2:
                    b = K.squeeze(b, self.sum_axes[2] - 2)
                if K.backend() == 'theano':
                    output += K.pattern_broadcast(b, b_broadcast)
                else:
                    output += b
            output = K.permute_dimensions(output, permute_dims)

        else:
            # bcast = list(np.where(self.broadcast)[0])
            permute_dims = list(range(ndim + 1))
            permute_dims[self.sum_axes[0]] = ndim
            permute_dims[ndim] = self.sum_axes[0]

            if K.backend() == 'theano':
                output = K.sum(x[..., None] * K.pattern_broadcast(W, self.kernel_broadcast), axis=self.sum_axes, keepdims=True)
            else:
                output = K.sum(x[..., None] * W, axis=self.sum_axes, keepdims=True)

            output_complex = K.sqrt(K.square(output[..., :self.filters_complex]) + K.square(output[..., self.filters_complex:2*self.filters_complex]) + K.epsilon())
            output_simple = output[..., 2*self.filters_complex:]
            output = K.concatenate([output_complex, output_simple], axis=-1)

            if self.use_bias:
                if K.backend() == 'theano':
                    output += K.pattern_broadcast(self.bias, self.bias_broadcast)
                else:
                    output += self.bias
            output = K.squeeze(K.permute_dimensions(output, permute_dims), ndim)
            if len(self.sum_axes) > 1:
                output = K.squeeze(output, self.sum_axes[1])

        return self.activation(output)

    def compute_output_shape(self, input_shape):

        ndim = len(input_shape)
        output_shape = [input_shape[0]] + [1] * (ndim-1)
        for i in set(range(1, ndim)) - set(self.sum_axes):
            output_shape[i] = input_shape[i]

        output_shape.append(self.filters_complex + self.filters_simple)
        permute_dims = list(range(ndim + 1))
        permute_dims[self.sum_axes[0]] = ndim
        permute_dims[ndim] = self.sum_axes[0]
        output_shape = [output_shape[i] for i in permute_dims]
        output_shape.pop(ndim)
        if len(self.sum_axes) > 1:
            output_shape.pop(self.sum_axes[1])


        return tuple(output_shape)

    def get_config(self):
        config = {
            'filters_simple': self.filters_simple,
            'filters_complex': self.filters_complex,
            'sum_axes': self.sum_axes,
            'filter_axes': self.filter_axes,
            'activation': activations.serialize(self.activation),
            'kernel_activation': activations.serialize(self.kernel_activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(FilterDimsV1, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class SoftMinMax(Layer):
    """Computes a selective and adaptive soft-min or soft-max over the inputs.

    # Example

    ```python
        # as first layer in a sequential model:
        model = Sequential()
        model.add(SoftMinMax(32, input_dim=16))
        # now the model will take as input arrays of shape (*, 16)
        # and output arrays of shape (*, 32)

        # this is equivalent to the above:
        model = Sequential()
        model.add(SoftMinMax(32, input_shape=(16,)))

        # after the first layer, you don't need to specify
        # the size of the input anymore:
        model.add(SoftMinMax(32))
    ```

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
        weights: list of Numpy arrays to set as initial weights.
            The list should have 2 elements, of shape `(input_dim, output_dim)`
            and (output_dim,) for weights and k parameters respectively.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        k_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the k parameters.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        k_constraint: instance of the [constraints](../constraints.md) module,
            applied to the k parameters.
        input_dim: dimensionality of the input (integer). This argument
            (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.

    # Input shape
        nD tensor with shape: `(nb_samples, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(nb_samples, input_dim)`.

    # Output shape
        nD tensor with shape: `(nb_samples, ..., output_dim)`.
        For instance, for a 2D input with shape `(nb_samples, input_dim)`,
        the output would have shape `(nb_samples, output_dim)`.
    """

    def __init__(self, units,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=constraints.NonNeg(),
                 k_initializer='zeros',
                 k_regularizer=None,
                 k_constraint=None,
                 tied_k=False,
                 activity_regularizer=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(SoftMinMax, self).__init__(**kwargs)

        self.units = units
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.k_initializer = initializers.get(k_initializer)
        self.k_regularizer = regularizers.get(k_regularizer)
        self.k_constraint = constraints.get(k_constraint)
        self.tied_k = tied_k
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        if self.tied_k:
            k_size = (1,)
        else:
            k_size = (self.units,)

        self.k = self.add_weight(shape=k_size,
                                 initializer=self.k_initializer,
                                 name='k',
                                 regularizer=self.k_regularizer,
                                 constraint=self.k_constraint)

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, x, mask=None):
        # W = K.softplus(10.*self.kernel)/10.
        W = self.kernel
        if self.tied_k:
            kX = self.k[0] * x
            kX = K.clip(kX, -30, 30)
            wekx = W[None, :, :] * K.exp(kX[:, :, None])
        else:
            kX = self.k[None, None, :] * x[:, :, None]
            kX = K.clip(kX, -30, 30)
            wekx = W[None, :, :] * K.exp(kX)
        output = K.sum(x[:, :, None] * wekx, axis=1) / (K.sum(wekx, axis=1) + K.epsilon())
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'k_initializer': initializers.serialize(self.k_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'k_regularizer': regularizers.serialize(self.k_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'k_constraint': constraints.serialize(self.k_constraint)
        }
        base_config = super(SoftMinMax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class WeightedMean(Layer):
    """Computes a selective and adaptive soft-min or soft-max over the inputs.

    # Example

    ```python
        # as first layer in a sequential model:
        model = Sequential()
        model.add(SoftMinMax(32, input_dim=16))
        # now the model will take as input arrays of shape (*, 16)
        # and output arrays of shape (*, 32)

        # this is equivalent to the above:
        model = Sequential()
        model.add(SoftMinMax(32, input_shape=(16,)))

        # after the first layer, you don't need to specify
        # the size of the input anymore:
        model.add(SoftMinMax(32))
    ```

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
        weights: list of Numpy arrays to set as initial weights.
            The list should have 2 elements, of shape `(input_dim, output_dim)`
            and (output_dim,) for weights and k parameters respectively.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        k_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the k parameters.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        k_constraint: instance of the [constraints](../constraints.md) module,
            applied to the k parameters.
        input_dim: dimensionality of the input (integer). This argument
            (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.

    # Input shape
        nD tensor with shape: `(nb_samples, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(nb_samples, input_dim)`.

    # Output shape
        nD tensor with shape: `(nb_samples, ..., output_dim)`.
        For instance, for a 2D input with shape `(nb_samples, input_dim)`,
        the output would have shape `(nb_samples, output_dim)`.
    """

    def __init__(self, units,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=constraints.NonNeg(),
                 activity_regularizer=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(WeightedMean, self).__init__(**kwargs)

        self.units = units
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, x, mask=None):
        # W = K.softplus(10.*self.kernel)/10.
        W = self.kernel
        wekx = W[None, :, :]
        output = K.sum(x[:, :, None] * wekx, axis=1) / (K.sum(wekx, axis=1) + K.epsilon())
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint)
        }
        base_config = super(WeightedMean, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DenseNonNeg(Layer):
    """A densely-connected NN layer with weights soft-rectified before 
    being applied.

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
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list of Numpy arrays to set as initial weights.
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
        bias: whether to include a bias
            (i.e. make the layer affine rather than linear).
        input_dim: dimensionality of the input (integer). This argument
            (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.

    # Input shape
        nD tensor with shape: `(nb_samples, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(nb_samples, input_dim)`.

    # Output shape
        nD tensor with shape: `(nb_samples, ..., output_dim)`.
        For instance, for a 2D input with shape `(nb_samples, input_dim)`,
        the output would have shape `(nb_samples, output_dim)`.
    """

    def __init__(self, output_dim, init='glorot_uniform',
                 activation=None, weights=None,
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
        self.input_spec = [InputSpec(ndim='2+')]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(DenseNonNeg, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.input_dim = input_dim
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     ndim='2+')]

        self.W = self.add_weight((input_dim, self.output_dim),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((self.output_dim,),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, x, mask=None):
        W = K.softplus(10.*self.W)/10.
        b = K.softplus(10.*self.b)/10.

        output = K.dot(x, W)
        if self.bias:
            output += b
        return self.activation(output)

    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1] and input_shape[-1] == self.input_dim
        output_shape = list(input_shape)
        output_shape[-1] = self.output_dim
        return tuple(output_shape)

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
        base_config = super(DenseNonNeg, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Feedback(Layer):
    """An adaptive Devisive Normalization layer where the output
    consists of the inputs added to a weighted combination of the inputs

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
        weights: list of Numpy arrays to set as initial weights.
            The list should have 2 elements, of shape `(input_dim, input_dim)`
            and (input_dim,) for weights and biases respectively.
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
        bias: whether to include a bias
            (i.e. make the layer affine rather than linear).
        input_dim: dimensionality of the input (integer). This argument
            (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.

    # Input shape
        nD tensor with shape: `(nb_samples, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(nb_samples, input_dim)`.

    # Output shape
        nD tensor with shape: `(nb_samples, ..., input_dim)`.
        For instance, for a 2D input with shape `(nb_samples, input_dim)`,
        the output would have shape `(nb_samples, input_dim)`.
    """

    def __init__(self, init='glorot_uniform',
                 activation=None, weights=None,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, input_dim=None, **kwargs):
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.input_dim = input_dim

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim='2+')]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(Feedback, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.input_dim = input_dim
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     ndim='2+')]

        self.W = self.add_weight((input_dim, input_dim),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_dim,),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, x, mask=None):
        output = K.dot(x, self.W)
        if self.bias:
            output += self.b
        return self.activation(x + output)

    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1] and input_shape[-1] == self.input_dim
        output_shape = list(input_shape)
        output_shape[-1] = self.input_dim
        return tuple(output_shape)

    def get_config(self):
        config = {'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'bias': self.bias,
                  'input_dim': self.input_dim}
        base_config = super(Feedback, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DivisiveNormalization(Layer):
    """An adaptive Devisive Normalization layer where the output
    consists of the inputs divided by a weighted combination of the inputs

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
        weights: list of Numpy arrays to set as initial weights.
            The list should have 2 elements, of shape `(input_dim, input_dim)`
            and (input_dim,) for weights and biases respectively.
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
        bias: whether to include a bias
            (i.e. make the layer affine rather than linear).
        input_dim: dimensionality of the input (integer). This argument
            (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.

    # Input shape
        nD tensor with shape: `(nb_samples, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(nb_samples, input_dim)`.

    # Output shape
        nD tensor with shape: `(nb_samples, ..., input_dim)`.
        For instance, for a 2D input with shape `(nb_samples, input_dim)`,
        the output would have shape `(nb_samples, input_dim)`.
    """

    def __init__(self, init='glorot_uniform',
                 activation=None, weights=None,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, input_dim=None, **kwargs):
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.input_dim = input_dim

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim='2+')]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(DivisiveNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.input_dim = input_dim
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     ndim='2+')]

        self.W = self.add_weight((input_dim, input_dim),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_dim,),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, x, mask=None):
        W = K.softplus(10.*self.W)/10.
        b = K.softplus(10.*self.b)/10.

        if self.bias:
            b = K.softplus(10.*self.b)/10.
            output = x / (1. + K.dot(x, W) + b)
        else:
            output = x / (1. + K.dot(x, W))
        return self.activation(output)

    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1] and input_shape[-1] == self.input_dim
        output_shape = list(input_shape)
        output_shape[-1] = self.input_dim
        return tuple(output_shape)

    def get_config(self):
        config = {'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'bias': self.bias,
                  'input_dim': self.input_dim}
        base_config = super(DivisiveNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Gram(Layer):

    def __init__(self, diag=True, input_dim=None, data_format=None, **kwargs):
        super(Gram, self).__init__(**kwargs)
        if data_format is None:
            data_format = K.image_data_format()
        self.data_format = data_format
        self.diag = diag
        self.input_dim = input_dim
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)

    def build(self, input_shape):
        ndim = len(input_shape)
        assert ndim == 4
        if self.data_format == 'channels_first':
            self.stack_size = input_shape[1]
        elif self.data_format == 'channels_last':
            self.stack_size = input_shape[3]
        else:
            raise ValueError('Invalid data_format:', self.data_format)

        if self.diag:
            d = 0
        else:
            d = -1

        self.tril = np.nonzero(np.tri(self.stack_size, self.stack_size, d).ravel())[0]

        self.built = True

    def call(self, x, mask=None):
        xshape = K.int_shape(x)
        if self.data_format == 'channels_first':
            x = K.reshape(x, [-1, xshape[1], xshape[2]*xshape[3]])
            out = K.batch_dot(x, K.permute_dimensions(x, [0, 2, 1]))
        elif self.data_format == 'channels_last':
            x = K.reshape(x, [-1, xshape[1]*xshape[2], xshape[3]])
            out = K.batch_dot(K.permute_dimensions(x, [0, 2, 1]), x)

        out = K.permute_dimensions(K.gather(K.permute_dimensions(K.reshape(out, [-1, self.stack_size**2]), [1, 0]), self.tril), [1, 0])
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], len(self.tril))

    def get_config(self):
        config = {
            'input_dim': self.input_dim}
        base_config = super(Gram, self).get_config()
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


class DenseDistance(Layer):
    """Just your regular densely-connected NN layer.
    `Dense` implements the operation:
    `output = activation(dot(input, kernel) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`).
    Note: if the input to the layer has a rank greater than 2, then
    it is flattened prior to the initial dot product with `kernel`.
    # Example
    ```python
        # as first layer in a sequential model:
        model = Sequential()
        model.add(Dense(32, input_shape=(16,)))
        # now the model will take as input arrays of shape (*, 16)
        # and output arrays of shape (*, 32)
        # after the first layer, you don't need to specify
        # the size of the input anymore:
        model.add(Dense(32))
    ```
    # Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
    # Input shape
        nD tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.
    # Output shape
        nD tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """

    def __init__(self, units,
                 activation=None,
                 L2square=False,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 metric='L2',
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(DenseDistance, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.metric = metric
        self.L2square = L2square
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        if self.metric is 'L1':
            return K.sum(K.abs(inputs[..., None] - self.kernel[None, ...]), axis=-2)
        elif self.L2square:
            return K.sum(K.square(inputs[..., None] - self.kernel[None, ...]), axis=-2)
        else:
            return K.sqrt(K.maximum(K.sum(K.square(inputs[..., None] - self.kernel[None, ...]), axis=-2), K.epsilon()))

        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint)
        }
        base_config = super(DenseDistance, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Distance2RBF(Layer):
    """Just your regular densely-connected NN layer.
    `Dense` implements the operation:
    `output = activation(dot(input, kernel) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`).
    Note: if the input to the layer has a rank greater than 2, then
    it is flattened prior to the initial dot product with `kernel`.
    # Example
    ```python
        # as first layer in a sequential model:
        model = Sequential()
        model.add(Dense(32, input_shape=(16,)))
        # now the model will take as input arrays of shape (*, 16)
        # and output arrays of shape (*, 32)
        # after the first layer, you don't need to specify
        # the size of the input anymore:
        model.add(Dense(32))
    ```
    # Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
    # Input shape
        nD tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.
    # Output shape
        nD tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """

    def __init__(self, units,
                 kernel_initializer='ones',
                 kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(DenseDistance, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.metric = metric
        self.L2square = L2square
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        if self.metric is 'L1':
            return K.sum(K.abs(inputs[..., None] - self.kernel[None, ...]), axis=-2)
        elif self.L2square:
            return K.sum(K.square(inputs[..., None] - self.kernel[None, ...]), axis=-2)
        else:
            return K.sqrt(K.maximum(K.sum(K.square(inputs[..., None] - self.kernel[None, ...]), axis=-2), K.epsilon()))

        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint)
        }
        base_config = super(DenseDistance, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Distance(Layer):
    """Just your regular densely-connected NN layer.
    `Dense` implements the operation:
    `output = activation(dot(input, kernel) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`).
    Note: if the input to the layer has a rank greater than 2, then
    it is flattened prior to the initial dot product with `kernel`.
    # Example
    ```python
        # as first layer in a sequential model:
        model = Sequential()
        model.add(Dense(32, input_shape=(16,)))
        # now the model will take as input arrays of shape (*, 16)
        # and output arrays of shape (*, 32)
        # after the first layer, you don't need to specify
        # the size of the input anymore:
        model.add(Dense(32))
    ```
    # Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
    # Input shape
        nD tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.
    # Output shape
        nD tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """

    def __init__(self,
                 metric='L2',
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(Distance, self).__init__(**kwargs)
        self.metric = metric
        self.input_spec = InputSpec(ndim=3)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.stack_size = input_shape[-2]
        self.tril = np.nonzero(np.tri(input_shape[-2], input_shape[-2], -1).ravel())[0]
        self.built = True

    def call(self, inputs):
        if self.metric is 'L1':
            out =  K.sum(K.abs(inputs[..., None] - K.permute_dimensions(inputs, (0,2,1))[:, None, ...]), axis=-2)
        elif self.metric is 'L2':
            out = K.sqrt(K.maximum(K.sum(K.square(inputs[..., None] - K.permute_dimensions(inputs, (0,2,1))[:, None, ...]), axis=-2), K.epsilon()))

        out = K.permute_dimensions(K.gather(K.permute_dimensions(K.reshape(out, [-1, self.stack_size**2]), [1, 0]), self.tril), [1, 0])
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], len(self.tril))

    def get_config(self):
        config = {
            'metric': self.metric,
        }
        base_config = super(Distance, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class GatedMultiply(Layer):
    """Just your regular densely-connected NN layer.
    `Dense` implements the operation:
    `output = activation(dot(input, kernel) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`).
    Note: if the input to the layer has a rank greater than 2, then
    it is flattened prior to the initial dot product with `kernel`.
    # Example
    ```python
        # as first layer in a sequential model:
        model = Sequential()
        model.add(Dense(32, input_shape=(16,)))
        # now the model will take as input arrays of shape (*, 16)
        # and output arrays of shape (*, 32)
        # after the first layer, you don't need to specify
        # the size of the input anymore:
        model.add(Dense(32))
    ```
    # Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
    # Input shape
        nD tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.
    # Output shape
        nD tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """

    def __init__(self, units,
                 kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(GatedMultiply, self).__init__(**kwargs)
        self.units = units
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.kernel_initializer = initializers.uniform(-.1, 0)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):

        output = K.exp(K.dot(K.log(inputs + 1e-5), K.sigmoid(100.*self.kernel)))

        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint)
        }
        base_config = super(GatedMultiply, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

