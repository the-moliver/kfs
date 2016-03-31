# -*- coding: utf-8 -*-
from __future__ import absolute_import

from keras import backend as K
from keras import activations, initializations, regularizers, constraints
from keras.layers.core import Layer


def conv_output_length(input_length, filter_size, border_mode, stride):
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - filter_size + 1
    return (output_length + stride - 1) // stride

class _Pooling2D_delay(Layer):
    '''Abstract class for different pooling 2D layers.
    '''
    input_ndim = 4

    def __init__(self, stack_size, pool_size=(2, 2), strides=None, border_mode='valid',
                 dim_ordering='th', **kwargs):
        super(_Pooling2D, self).__init__(**kwargs)
        self.input = K.placeholder(ndim=4)
        self.pool_size = tuple(pool_size)
        self.stack_size = stack_size
        if strides is None:
            strides = self.pool_size
        self.strides = tuple(strides)
        assert border_mode in {'valid', 'same'}, 'border_mode must be in {valid, same}'
        self.border_mode = border_mode
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering

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

        rows = conv_output_length(rows, self.pool_size[0],
                                  self.border_mode, self.strides[0])
        cols = conv_output_length(cols, self.pool_size[1],
                                  self.border_mode, self.strides[1])

        if self.dim_ordering == 'th':
            return (input_shape[0], input_shape[1], rows, cols)
        elif self.dim_ordering == 'tf':
            return (input_shape[0], rows, cols, input_shape[3])
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def _pooling_function(self, inputs, pool_size, strides,
                          border_mode, dim_ordering):
        raise NotImplementedError

    def get_output(self, train=False):
        X = self.get_input(train)
        samps = X.shape[0]
        maxlen = X.shape[1]
        dim = K.cast(K.sqrt(X.shape[2]/self.stack_size), 'int64')
        X = K.reshape(X, (samps*maxlen, self.stack_size, dim, dim))
        output = self._pooling_function(inputs=X, pool_size=self.pool_size,
                                        strides=self.strides,
                                        border_mode=self.border_mode,
                                        dim_ordering=self.dim_ordering)
        output = K.reshape(output, (samps, maxlen, output.shape[1]*output.shape[2]*output.shape[3]))
        return output

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'pool_size': self.pool_size,
                  'border_mode': self.border_mode,
                  'strides': self.strides,
                  'dim_ordering': self.dim_ordering}
        base_config = super(_Pooling2D_delay, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AveragePooling2D_delay(_Pooling2D_delay):
    '''Average pooling operation for spatial data.

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.

    # Output shape
        4D tensor with shape:
        `(nb_samples, channels, pooled_rows, pooled_cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, pooled_rows, pooled_cols, channels)` if dim_ordering='tf'.

    # Arguments
        pool_size: tuple of 2 integers,
            factors by which to downscale (vertical, horizontal).
            (2, 2) will halve the image in each dimension.
        strides: tuple of 2 integers, or None. Strides values.
        border_mode: 'valid' or 'same'.
            Note: 'same' will only work with TensorFlow for the time being.
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode is it at index 3.
    '''
    def __init__(self, pool_size=(2, 2), strides=None, border_mode='valid',
                 dim_ordering='th', **kwargs):
        super(AveragePooling2D_delay, self).__init__(pool_size, strides, border_mode,
                                               dim_ordering, **kwargs)

    def _pooling_function(self, inputs, pool_size, strides,
                          border_mode, dim_ordering):
        output = K.pool2d(inputs, pool_size, strides,
                          border_mode, dim_ordering, pool_mode='avg')
        return output


class Convolution2DColorEnergy_DelayBasis2(Layer):
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
        `(samples, nb_filter, new_rows, new_cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, nb_filter)` if dim_ordering='tf'.
        `rows` and `cols` values might have changed due to padding.


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

    def __init__(self, nb_filter, nb_row, nb_col, delays, delay_basis,
                 init='glorot_uniform', activation='linear', weights=None,
                 border_mode='valid', subsample=(1, 1), dim_ordering='th',
                 W1_regularizer=None, W2_regularizer=None, Wlin_regularizer=None, Wt_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W1_constraint=None, W2_constraint=None, Wlin_constraint=None, Wt_constraint=None, b_constraint=None, **kwargs):

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

        self.W1_regularizer = regularizers.get(W1_regularizer)
        self.W2_regularizer = regularizers.get(W2_regularizer)
        self.Wlin_regularizer = regularizers.get(Wlin_regularizer)
        self.Wt_regularizer = regularizers.get(Wt_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W1_constraint = constraints.get(W1_constraint)
        self.W2_constraint = constraints.get(W2_constraint)
        self.Wlin_constraint = constraints.get(Wlin_constraint)
        self.Wt_constraint = constraints.get(Wt_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.constraints = [self.W1_constraint, self.W2_constraint, self.Wlin_constraint, self.Wt_constraint, self.b_constraint]

        self.initial_weights = weights
        self.input = K.placeholder(ndim=4)
        super(Convolution2D, self).__init__(**kwargs)

    def build(self):
        if self.dim_ordering == 'th':
            stack_size = self.input_shape[1]
            self.W_shape = (self.nb_filter, 3, self.nb_row, self.nb_col)
            self.Wlin_shape = (1, stack_size, self.nb_row, self.nb_col)
        elif self.dim_ordering == 'tf':
            stack_size = self.input_shape[3]
            self.W_shape = (self.nb_row, self.nb_col, 3, self.nb_filter)
            self.Wlin_shape = (1, self.nb_col, 3, self.nb_filter)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        self.W1 = self.init(self.W_shape)
        self.W2 = self.init(self.W_shape)
        self.Wlin = self.init(self.Wlin_shape)
        self.Wt = self.init((delays, delay_basis))
        self.b = K.zeros((self.nb_filter,))
        self.params = [self.W1, self.W2, self.Wlin, self.Wt, self.b]
        self.regularizers = []

        if self.W1_regularizer:
            self.W1_regularizer.set_param(self.W1)
            self.regularizers.append(self.W1_regularizer)

        if self.W2_regularizer:
            self.W2_regularizer.set_param(self.W2)
            self.regularizers.append(self.W2_regularizer)

        if self.Wlin_regularizer:
            self.Wlin_regularizer.set_param(self.Wlin)
            self.regularizers.append(self.Wlin_regularizer)

        if self.Wt_regularizer:
            self.Wt_regularizer.set_param(self.Wt)
            self.regularizers.append(self.Wt_regularizer)

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
        samps = X.shape[0]
        maxlen = X.shape[1]
        dim = K.cast(K.sqrt(X.shape[2]/3), 'int64')
        X = K.reshape(X, (samps*maxlen, 3, dim, dim))
        conv_out1 = K.conv2d(X, self.W1, strides=self.subsample,
                            border_mode=self.border_mode,
                            dim_ordering=self.dim_ordering,
                            image_shape=self.input_shape,
                            filter_shape=self.W1_shape)

        conv_out2 = K.conv2d(X, self.W2, strides=self.subsample,
                            border_mode=self.border_mode,
                            dim_ordering=self.dim_ordering,
                            image_shape=self.input_shape,
                            filter_shape=self.W2_shape)

        conv_out3 = K.conv2d(X, self.Wlin, strides=self.subsample,
                            border_mode=self.border_mode,
                            dim_ordering=self.dim_ordering,
                            image_shape=self.input_shape,
                            filter_shape=self.Wlin_shape)

        conv_out1 = K.reshape(conv_out1, (samps, maxlen, conv_out1.shape[1], conv_out1.shape[2]*conv_out1.shape[3]))
        conv_out2 = K.reshape(conv_out2, (samps, maxlen, conv_out2.shape[1], conv_out2.shape[2]*conv_out2.shape[3]))
        conv_out3 = K.reshape(conv_out3, (samps, maxlen, conv_out3.shape[1], conv_out3.shape[2]*conv_out3.shape[3]))

        conv_out3 = K.dot(conv_out3.dimshuffle(0,2,3,1), self.Wt).dimshuffle(0,3,1,2)

        conv_out = K.sqrt(K.square(K.dot(conv_out1.dimshuffle(0,2,3,1), self.Wt)) + K.square(K.dot(conv_out2.dimshuffle(0,2,3,1), self.Wt))+K.epsilon()).dimshuffle(0,3,1,2)

        if self.dim_ordering == 'th':
            conv_out = conv_out + K.reshape(self.b, (1, self.nb_filter, 1, 1))
        elif self.dim_ordering == 'tf':
            conv_out = conv_out + K.reshape(self.b, (1, 1, 1, self.nb_filter))
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        output = K.concatenate([conv_out, conv_out3], axis=2)
        return K.reshape(output, (samps, self.delay_basis, output.shape[2]*output.shape[3]))

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
                  'Wlin_regularizer': self.Wlin_regularizer.get_config() if self.Wlin_regularizer else None,
                  'Wt_regularizer': self.Wt_regularizer.get_config() if self.Wt_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W1_constraint': self.W1_constraint.get_config() if self.W1_constraint else None,
                  'W2_constraint': self.W2_constraint.get_config() if self.W2_constraint else None,
                  'Wlin_constraint': self.Wlin_constraint.get_config() if self.Wlin_constraint else None,
                  'Wt_constraint': self.Wt_constraint.get_config() if self.Wt_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None}
        base_config = super(Convolution2D, self).get_config()
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
        `(samples, nb_filter, new_rows, new_cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, nb_filter)` if dim_ordering='tf'.
        `rows` and `cols` values might have changed due to padding.


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

    def __init__(self, nb_filter, nb_row, nb_col, stack_size,
                 init='glorot_uniform', activation='linear', weights=None,
                 border_mode='valid', subsample=(1, 1), dim_ordering='th',
                 W1_regularizer=None, W2_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W1_constraint=None, W2_constraint=None, b_constraint=None, **kwargs):

        if border_mode not in {'valid', 'same'}:
            raise Exception('Invalid border mode for Convolution2D:', border_mode)
        self.nb_filter = nb_filter
        self.nb_row = nb_row
        self.nb_col = nb_col
        self.stack_size = stack_size
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        assert border_mode in {'valid', 'same'}, 'border_mode must be in {valid, same}'
        self.border_mode = border_mode
        self.subsample = tuple(subsample)
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering

        self.W1_regularizer = regularizers.get(W1_regularizer)
        self.W2_regularizer = regularizers.get(W2_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W1_constraint = constraints.get(W1_constraint)
        self.W2_constraint = constraints.get(W2_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.constraints = [self.W1_constraint, self.W2_constraint, self.b_constraint]

        self.initial_weights = weights
        self.input = K.placeholder(ndim=4)
        super(Convolution2DEnergy, self).__init__(**kwargs)

    def build(self):
        if self.dim_ordering == 'th':

            self.W_shape = (self.nb_filter, 1, self.nb_row, self.nb_col)
        elif self.dim_ordering == 'tf':

            self.W_shape = (self.nb_row, self.nb_col, 1, self.nb_filter)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        self.W1 = self.init(self.W_shape)
        self.W2 = self.init(self.W_shape)
        self.b = K.zeros((self.nb_filter,))
        self.params = [self.W1, self.W2, self.b]
        self.regularizers = []

        if self.W1_regularizer:
            self.W1_regularizer.set_param(self.W1)
            self.regularizers.append(self.W1_regularizer)

        if self.W2_regularizer:
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
            return (input_shape[0], self.nb_filter*self.stack_size, rows, cols)
        elif self.dim_ordering == 'tf':
            return (input_shape[0], rows, cols, self.nb_filter*self.stack_size)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def get_output(self, train=False):
        X = self.get_input(train)

        if self.dim_ordering == 'th':
            samps = X.shape[0]
            maxlen = X.shape[1]
            X = K.reshape(X, (samps*maxlen, 1, X.shape[2], X.shape[3]))
            axis=(2,3)
        elif self.dim_ordering == 'tf':
            samps = X.shape[0]
            maxlen = X.shape[3]
            X = K.reshape(X.dimshuffle(0,3,1,2), (samps*maxlen, X.shape[2], X.shape[3], 1))
            axis=(0,1)
        
        if train:
            W1 = self.W1
            W2 = self.W2 - W1*K.sum(W1*self.W2, axis=axis, keepdims=True)/K.sum(W1*W1, axis=axis, keepdims=True)
            norm = K.sqrt(K.sum(W1**2 + W2**2, axis=axis, keepdims=True))
            self.W1 = W1 / norm
            self.W2 = W2 / norm
   
        conv_out1 = K.conv2d(X, self.W1, strides=self.subsample,
                            border_mode=self.border_mode,
                            dim_ordering=self.dim_ordering,
                            image_shape=self.input_shape,
                            filter_shape=self.W_shape)
        conv_out2 = K.conv2d(X, self.W2, strides=self.subsample,
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

        if self.dim_ordering == 'th':
            output = K.reshape(output, (samps, maxlen*conv_out.shape[1], conv_out.shape[2], conv_out.shape[3]))
        elif self.dim_ordering == 'tf':
            output = K.reshape(output, (samps, maxlen, conv_out.shape[1], conv_out.shape[2], conv_out.shape[3])).dimshuffle(0,2,3,1,4)
            output = K.reshape(output, (conv_out.shape[0], conv_out.shape[1], conv_out.shape[2], conv_out.shape[3]*conv_out.shape[4]))
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
                  'W1_constraint': self.W1_constraint.get_config() if self.W1_constraint else None,
                  'W2_constraint': self.W2_constraint.get_config() if self.W2_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None}
        base_config = super(Convolution2DEnergy, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

