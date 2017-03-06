# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division

import functools

from keras import backend as K
from keras import activations, initializers, regularizers, constraints
from keras.engine import InputSpec, Layer


class TemporalFilter(Layer):
    def __init__(self, output_dim, init='glorot_uniform', activation='linear', weights=None,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None, input_dim=None, bias=False, **kwargs):
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.bias = bias

        self.W_regularizer = regularizers.get(W_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)

        if self.bias:
            self.b_regularizer = regularizers.get(b_regularizer)
            self.b_constraint = constraints.get(b_constraint)

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
                  'input_dim': self.input_dim}
        base_config = super(TemporalFilter, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SpatialFilter(Layer):
    def __init__(self, output_dim, init='glorot_uniform', activation='linear', weights=None,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None, bias=False,
                 W_constraint=None, b_constraint=None, input_dim=None, **kwargs):
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.bias = bias

        self.W_regularizer = regularizers.get(W_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)

        if self.bias:
            self.b_regularizer = regularizers.get(b_regularizer)
            self.b_constraint = constraints.get(b_constraint)

        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=5)]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(SpatialFilter, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 5
        input_dim1 = input_shape[1]
        input_dim2 = input_shape[2]
        input_dim3 = input_shape[3]
        input_dim4 = input_shape[4]

        self.W = self.init((input_dim3, input_dim4, self.output_dim),
                           name='{}_W'.format(self.name))
        if self.bias:
            self.b = K.zeros((input_dim1, input_dim2, self.output_dim),
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
        samps = x.shape[0]
        maxlen = x.shape[1]
        stack_size = x.shape[2]
        dim1 = x.shape[3]
        dim2 = x.shape[4]

        x = K.reshape(x, (samps*maxlen*stack_size, dim1*dim2))
        W = K.reshape(self.W, (dim1*dim2, self.output_dim))
        output = K.reshape(K.dot(x, W), (samps, maxlen, stack_size, self.output_dim))
        if self.bias:
            output += self.b
        return self.activation(output)

    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) == 5
        return (input_shape[0], input_shape[1], input_shape[2], self.output_dim)

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'input_dim': self.input_dim}
        base_config = super(SpatialFilter, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


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

        model.add(FilterDims(nb_filters=5, sum_axes=[1], filter_axes=[1], bias=False))

        # The weights learned by FilterDims are a set of temporal filters on the output of the spatial convolutions
        # The output dimensionality is (#samples, 5, 10, 13, 13)
        # We can then use FilterDims to filter the 5 temporal dimensions and 10 convolutional filter feature map
        # dimensions to create 2 spatio-temporal filters with a 5x10x2 weight tensor:

        model.add(FilterDims(nb_filters=2, sum_axes=[1, 2], filter_axes=[1, 2], bias=False))

        # The output dimensionality is (#samples, 2, 13, 13)
        # We can then use FilterDims to spatially filter each spatio-temporal dimension with a 2x13x13 tensor:

        model.add(FilterDims(nb_filters=1, sum_axes=[2, 3], filter_axes=[1, 2, 3], bias=False))

        # We only sum over the last two spatial axes resutling in an output dimensionality of (#samples, 2)
    ```

    # Arguments
        nb_filters: number of filters to apply.
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
    def __init__(self, nb_filters, sum_axes, filter_axes, init='glorot_uniform', activation='linear', weight_activation='linear',
                 weights=None, W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, input_dim=None, **kwargs):
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.weight_activation = activations.get(weight_activation)
        self.nb_filters = nb_filters
        self.sum_axes = list(sum_axes)
        self.sum_axes.sort()
        self.filter_axes = list(filter_axes)
        self.filter_axes.sort()
        self.input_dim = input_dim

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.initial_weights = weights

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(FilterDims, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        ndim = len(input_shape)

        w_shape = [1] * (ndim - 1)
        W_broadcast = [False] * (ndim - 1)
        b_broadcast = [True] * (ndim - 1)
        for i in self.filter_axes:
            w_shape[i-1] = input_shape[i]

        if self.nb_filters > 1:
            w_shape.append(self.nb_filters)
            W_broadcast.append(False)
            b_broadcast.append(False)

        for i in set(range(1, ndim)) - set(self.filter_axes):
            W_broadcast[i-1] = True

        self.W = self.add_weight(w_shape,
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)

        if self.bias:
            bias_size = [1] * (ndim - 1)
            for i in set(self.filter_axes) - set(self.sum_axes):
                bias_size[i-1] = input_shape[i]
                b_broadcast[i-1] = False

            if self.nb_filters > 1:
                bias_size.append(self.nb_filters)

            self.b = self.add_weight(bias_size,
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.W_broadcast = W_broadcast
        self.b_broadcast = b_broadcast
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, x, mask=None):
        ndim = len(self.input_spec[0].shape)
        W = self.weight_activation(self.W)

        if self.filter_axes == self.sum_axes:
            ax1 = [a-1 for a in self.sum_axes]
            ax1 = ax1 + list(set(range(ndim)) - set(ax1))
            ax2 = list(set(range(ndim)) - set(self.sum_axes))
            permute_dims = range(len(ax2))
            permute_dims.insert(self.sum_axes[0], len(ax2))
            outdims = [-1] + [self.input_spec[0].shape[a] for a in ax2[1:]] + [self.nb_filters]
            ax2 = ax2 + self.sum_axes
            W = K.permute_dimensions(W, ax1)
            W = K.reshape(W, (-1, self.nb_filters))
            x = K.permute_dimensions(x, ax2)
            x = K.reshape(x, (-1, K.shape(W)[0]))
            output = K.reshape(K.dot(x, W), outdims)
            if self.bias:
                b_broadcast = [i for j, i in enumerate(self.b_broadcast) if j not in self.sum_axes]
                b = K.squeeze(self.b, self.sum_axes[0])
                if len(self.sum_axes) > 1:
                    b = K.squeeze(b, self.sum_axes[1] - 1)
                if K.backend() == 'theano':
                    output += K.pattern_broadcast(b, b_broadcast)
                else:
                    output += b
            output = K.permute_dimensions(output, permute_dims)

        elif self.nb_filters > 1:
            # bcast = list(np.where(self.broadcast)[0])
            permute_dims = range(ndim + 1)
            permute_dims[self.sum_axes[0]] = ndim
            permute_dims[ndim] = self.sum_axes[0]

            if K.backend() == 'theano':
                output = K.sum(x[..., None] * K.pattern_broadcast(W, self.W_broadcast), axis=self.sum_axes, keepdims=True)
            else:
                output = K.sum(x[..., None] * W, axis=self.sum_axes, keepdims=True)

            if self.bias:
                if K.backend() == 'theano':
                    output += K.pattern_broadcast(self.b, self.b_broadcast)
                else:
                    output += self.b
            output = K.squeeze(K.permute_dimensions(output, permute_dims), ndim)
            if len(self.sum_axes) > 1:
                output = K.squeeze(output, self.sum_axes[1])

        else:
            if K.backend() == 'theano':
                output = K.sum(x * K.pattern_broadcast(W, self.W_broadcast), axis=self.sum_axes, keepdims=True)
            else:
                output = K.sum(x * W, axis=self.sum_axes, keepdims=True)
            if self.bias:
                if K.backend() == 'theano':
                    output += K.pattern_broadcast(self.b, self.b_broadcast)
                else:
                    output += self.b
            output = K.squeeze(output, self.sum_axes[0])
            if len(self.sum_axes) > 1:
                output = K.squeeze(output, self.sum_axes[1]-1)

        return self.activation(output)

    def get_output_shape_for(self, input_shape):
        if self.nb_filters > 1:
            ndim = len(input_shape)
            output_shape = [input_shape[0]] + [1] * (ndim-1)
            for i in set(range(1, ndim)) - set(self.sum_axes):
                output_shape[i] = input_shape[i]

            output_shape.append(self.nb_filters)
            permute_dims = range(ndim + 1)
            permute_dims[self.sum_axes[0]] = ndim
            permute_dims[ndim] = self.sum_axes[0]
            output_shape = [output_shape[i] for i in permute_dims]
            output_shape.pop(ndim)
            if len(self.sum_axes) > 1:
                output_shape.pop(self.sum_axes[1])
        else:
            output_shape = input_shape
            output_shape = [output_shape[i] for i in set(range(len(input_shape))) - set(self.sum_axes)]

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
        base_config = super(FilterDims, self).get_config()
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

    def __init__(self, output_dim, init='glorot_uniform',
                 activation=None, weights=None,
                 W_regularizer=None, k_regularizer=None, activity_regularizer=None,
                 W_constraint=None, k_constraint=None,
                 input_dim=None, **kwargs):
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.W_regularizer = regularizers.get(W_regularizer)
        self.k_regularizer = regularizers.get(k_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.k_constraint = constraints.get(k_constraint)

        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim='2+')]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(SoftMinMax, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[-1]
        self.input_dim = input_dim
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     ndim='2+')]

        self.W = self.add_weight((input_dim, self.output_dim),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)

        self.k = self.add_weight((self.output_dim,),
                                 initializer='zero',
                                 name='{}_k'.format(self.name),
                                 regularizer=self.k_regularizer,
                                 constraint=self.k_constraint)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, x, mask=None):
        W = K.softplus(10.*self.W)/10.
        kX = self.k[None, None, :] * x[:, :, None]
        kX = K.clip(kX, -30, 30)
        wekx = W[None, :, :] * K.exp(kX)
        output = K.sum(x[:, :, None] * wekx, axis=1) / (K.sum(wekx, axis=1) + 1)
        return output

    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) == 2
        assert input_shape[-1] and input_shape[-1] == self.input_dim
        output_shape = list(input_shape)
        output_shape[-1] = self.output_dim
        return tuple(output_shape)

    def get_config(self):
        config = {'output_dim': self.output_dim,
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