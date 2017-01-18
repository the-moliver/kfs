# -*- coding: utf-8 -*-
from __future__ import absolute_import
import functools

from keras import backend as K
from keras import activations, initializations, regularizers
from kfs import constraints
from keras.engine import Layer, InputSpec
import numpy as np


def conv_output_length(input_length, filter_size, border_mode, stride):
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - filter_size + 1
    return (output_length + stride - 1) // stride


def step_init(params, name):
    step = 2./(params[0]-1.)
    steps = params[1]*(np.arange(0, 2+step, step)[:params[0]] - 1.)
    return K.variable(steps, name=name)


class Convolution2DEnergy_TemporalBasis(Layer):
    """Convolution operator for filtering windows of time varying
    two-dimensional inputs, such as a series of movie frames, with
    learned filters inspired by simple-cell and complex-cell V1 neurons.
    Filters are learned in a factorized representation, consisting of
    orthogonal 2D filters, a set of vectors that control filter amplitude
    over time, and a set of scalars that control the trade-off of the
    orthogonal 2D filters over time. This representation can create a large
    number of 3D spatio-temporal filters from a small number of parameters,
    often with less than 1% of the parameters of a naive 3D convolutional
    model. When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(12, 3, 64, 64)` for 12 64x64 RGB pictures.
    # Examples
    ```python
        # apply a 5x5 convolution with 8 simple-cell filters and
        16 complex-cell filters with 4 amplitude profiles and 7
        temporal frequencies on a 256x256 image:
        model = Sequential()
        model.add(Convolution2D(8, 16, 5, 5, 4, 7,
                                border_mode='same',
                                input_shape=(3, 256, 256)))
        # now model.output_shape == (None, 4*7, 8+16, 256, 256)
    ```
    # Arguments
        nb_filter_simple: Number of simple-cell filters to use.
        nb_filter_complex: Number of complex-cell filters to use.
        nb_row: Number of rows in the convolution kernel.
        nb_col: Number of columns in the convolution kernel.
        nb_temporal_amplitude: Number of amplitude profiles
        nb_temporal_freq: Number of temporal frequencies (odd number)
        tf_max: Maximum temporal frequency, temporal frequencies initialized
                as (-tf_max..., 0, ..., tf_max)
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
        border_mode: 'valid', 'same' or 'full'
            ('full' requires the Theano backend).
        subsample: tuple of length 2. Factor by which to subsample output.
            Also called strides elsewhere.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        Wt_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L2 or Laplacian regularization), applied to the temporal amplitude
            weights matrix.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        Wt_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        b_constraint: instance of the [constraints](../constraints.md) module,
            applied to the bias.
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode is it at index 3.
            It defaults to the `image_dim_ordering` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "tf".
        bias: whether to include a bias
            (i.e. make the layer affine rather than linear).
    # Input shape
        5D tensor with shape:
        `(samples, time_steps, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, time_steps, rows, cols, channels)` if dim_ordering='tf'.
    # Output shape
        5D tensor with shape:
        `(samples, nb_temporal_filter, nb_filter, new_rows, new_cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, nb_temporal_filter, new_rows, new_cols, nb_filter)` if dim_ordering='tf'.
        `rows` and `cols` values might have changed due to padding.
    """

    def __init__(self, nb_filter_simple, nb_filter_complex, nb_row, nb_col, nb_temporal_amplitude,
                 nb_temporal_freq, tf_max=2, init='glorot_uniform', activation='relu', weights=None,
                 border_mode='valid', subsample=(1, 1), dim_ordering=K.image_dim_ordering(),
                 W_regularizer=None, Wt_regularizer=None, Ws_regularizer=None, b_regularizer=None,
                 activity_regularizer=None, Wt_constraint=None, Ws_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        if border_mode not in {'valid', 'same'}:
            raise Exception('Invalid border mode for Convolution2D:', border_mode)
        self.nb_filter_simple = nb_filter_simple
        self.nb_filter_complex = nb_filter_complex
        self.nb_row = nb_row
        self.nb_col = nb_col
        self.nb_temporal_amplitude = nb_temporal_amplitude
        self.nb_temporal_freq = nb_temporal_freq
        self.init = initializations.get(init, dim_ordering=dim_ordering)
        self.activation = activations.get(activation)
        assert border_mode in {'valid', 'same'}, 'border_mode must be in {valid, same}'
        self.border_mode = border_mode
        self.subsample = tuple(subsample)
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering
        self.tf_max = tf_max

        self.W_regularizer = regularizers.get(W_regularizer)
        self.Wt_regularizer = regularizers.get(Wt_regularizer)
        self.Ws_regularizer = regularizers.get(Ws_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.UnitNormOrthogonal(self.nb_filter_complex+self.nb_filter_simple)
        self.Wt_constraint = constraints.get(Wt_constraint)
        self.Ws_constraint = constraints.get(Ws_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.input_spec = [InputSpec(ndim=5)]
        self.initial_weights = weights
        super(Convolution2DEnergy_TemporalBasis, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 5
        self.input_spec = [InputSpec(shape=input_shape)]
        delays = input_shape[1]

        if self.dim_ordering == 'th':
            stack_size = input_shape[2]
            self.W_shape = (2*self.nb_filter_complex + 2*self.nb_filter_simple, stack_size, self.nb_row, self.nb_col)
        elif self.dim_ordering == 'tf':
            stack_size = input_shape[4]
            self.W_shape = (self.nb_row, self.nb_col, stack_size, 2*self.nb_filter_complex + 2*self.nb_filter_simple)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        self.W = self.add_weight(self.W_shape,
                                 initializer=functools.partial(self.init,
                                                               dim_ordering=self.dim_ordering),
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)

        self.Wt = self.add_weight((delays, self.nb_temporal_amplitude),
                                  initializer=functools.partial(self.init,
                                                                dim_ordering=self.dim_ordering),
                                  name='{}_Wt'.format(self.name),
                                  regularizer=self.Wt_regularizer,
                                  constraint=self.Wt_constraint)

        if self.bias:
            self.b = self.add_weight((self.nb_filter_complex + self.nb_filter_simple,),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.Ws = self.add_weight((self.nb_temporal_freq, self.tf_max),
                                  initializer=step_init,
                                  name='{}_Ws'.format(self.name),
                                  regularizer=self.Ws_regularizer,
                                  constraint=self.Ws_constraint)

        self.delays_pi = K.variable(2*np.pi*np.arange(0, 1+1./(delays-1), 1./(delays-1)), name='{}_delays_pi'.format(self.name))

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def get_output_shape_for(self, input_shape):
        if self.dim_ordering == 'th':
            conv_dim2 = input_shape[3]
            conv_dim3 = input_shape[4]
        elif self.dim_ordering == 'tf':
            conv_dim2 = input_shape[2]
            conv_dim3 = input_shape[3]
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        conv_dim2 = conv_output_length(conv_dim2, self.nb_row,
                                       self.border_mode, self.subsample[0])
        conv_dim3 = conv_output_length(conv_dim3, self.nb_col,
                                       self.border_mode, self.subsample[1])

        if self.dim_ordering == 'th':
            return (input_shape[0], self.nb_temporal_amplitude*self.nb_temporal_freq, (self.nb_filter_complex + self.nb_filter_simple), conv_dim2, conv_dim3)
        elif self.dim_ordering == 'tf':
            return (input_shape[0], self.nb_temporal_amplitude*self.nb_temporal_freq, conv_dim2, conv_dim3, (self.nb_filter_complex + self.nb_filter_simple))
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def call(self, x, mask=None):
        # input_shape = self.input_spec[0].shape
        # output_shape = [-1] + list(self.get_output_shape_for(input_shape)[1:])
        xshape = K.shape(x)
        x = K.reshape(x, (-1, xshape[2], xshape[3], xshape[4]))

        sin_step = K.reshape(K.sin(self.delays_pi[:, None, None]*self.Ws[None, :, None])*self.Wt[:, None, :], (-1, self.nb_temporal_amplitude*self.nb_temporal_freq))
        cos_step = K.reshape(K.cos(self.delays_pi[:, None, None]*self.Ws[None, :, None])*self.Wt[:, None, :], (-1, self.nb_temporal_amplitude*self.nb_temporal_freq))

        w0t = K.concatenate((cos_step, -sin_step), axis=0)
        w1t = K.concatenate((sin_step, cos_step), axis=0)

        conv_out1 = K.conv2d(x, self.W, strides=self.subsample,
                             border_mode=self.border_mode,
                             dim_ordering=self.dim_ordering,
                             filter_shape=self.W_shape)

        conv_out1 = K.reshape(conv_out1, (xshape[0], xshape[1], conv_out1.shape[1], conv_out1.shape[2], conv_out1.shape[3]))

        if self.dim_ordering == 'th':
            # samps x delays x stack x X x Y
            conv_out1 = K.permute_dimensions(conv_out1, (0, 2, 3, 4, 1))
            # samps x stack x X x Y x delays
        elif self.dim_ordering == 'tf':
            # samps x delays x X x Y x stack
            conv_out1 = K.permute_dimensions(conv_out1, (0, 4, 2, 3, 1))
            # samps x stack x X x Y x delays
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        # split out complex and simple filter pairs
        conv_out12 = K.concatenate([conv_out1[:, :self.nb_filter_complex, :, :, :], conv_out1[:, self.nb_filter_complex+self.nb_filter_simple:2*self.nb_filter_complex+self.nb_filter_simple, :, :, :]], axis=4)
        conv_out34 = K.concatenate([conv_out1[:, self.nb_filter_complex:self.nb_filter_complex + self.nb_filter_simple, :, :, :], conv_out1[:, 2*self.nb_filter_complex + self.nb_filter_simple:, :, :, :]], axis=4)

        # apply temporal trade-off to get temporal filter outputs and compute complex and simple outputs
        conv_out = K.sqrt(K.square(K.dot(conv_out12, w0t)) + K.square(K.dot(conv_out12, w1t)) + K.epsilon())
        conv_outlin = K.dot(conv_out34, w0t)
        # samps x stack x X x Y x nb_temporal_amplitude*nb_temporal_freq

        output = K.concatenate([conv_out, conv_outlin], axis=1)

        if self.dim_ordering == 'th':
            output = K.permute_dimensions(output, (0, 4, 1, 2, 3))
            # samps x nb_temporal_amplitude*nb_temporal_freq x stack x X x Y
        elif self.dim_ordering == 'tf':
            output = K.permute_dimensions(output, (0, 4, 2, 3, 1))
            # samps x nb_temporal_amplitude*nb_temporal_freq x X x Y x stack

        if self.bias:
            if self.dim_ordering == 'th':
                output += K.reshape(self.b, (1, 1, self.nb_filter_complex + self.nb_filter_simple, 1, 1))
            elif self.dim_ordering == 'tf':
                output += K.reshape(self.b, (1, 1, 1, 1, self.nb_filter_complex + self.nb_filter_simple))

        # output = K.reshape(self.activation(output), output_shape)
        output = self.activation(output)
        return output

    def get_config(self):
        config = {'nb_filter_simple': self.nb_filter_simple,
                  'nb_filter_complex': self.nb_filter_complex,
                  'nb_row': self.nb_row,
                  'nb_col': self.nb_col,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'border_mode': self.border_mode,
                  'subsample': self.subsample,
                  'dim_ordering': self.dim_ordering,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'Wt_regularizer': self.Wt_regularizer.get_config() if self.Wt_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'Wt_constraint': self.Wt_constraint.get_config() if self.Wt_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'bias': self.bias}
        base_config = super(Convolution2DEnergy_TemporalBasis, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Convolution2DEnergy_Scatter(Layer):
    def __init__(self, nb_filter_simple, nb_filter_complex, nb_row, nb_col,
                 init='glorot_uniform', activation='relu', weights=None,
                 border_mode='valid', subsample=(1, 1), dim_ordering=K.image_dim_ordering(),
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        if border_mode not in {'valid', 'same'}:
            raise Exception('Invalid border mode for Convolution2DEnergy_Scatter:', border_mode)
        self.nb_filter_simple = nb_filter_simple
        self.nb_filter_complex = nb_filter_complex
        self.nb_row = nb_row
        self.nb_col = nb_col
        self.init = initializations.get(init, dim_ordering=dim_ordering)
        self.activation = activations.get(activation)
        assert border_mode in {'valid', 'same'}, 'border_mode must be in {valid, same}'
        self.border_mode = border_mode
        self.subsample = tuple(subsample)
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.UnitNormOrthogonal(nb_filter_complex, dim_ordering)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.input_spec = [InputSpec(ndim=4)]
        self.initial_weights = weights
        super(Convolution2DEnergy_Scatter, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 4
        self.input_spec = [InputSpec(shape=input_shape)]

        if self.dim_ordering == 'th':
            self.W_shape = (2*self.nb_filter_complex + self.nb_filter_simple, 1, self.nb_row, self.nb_col)
        elif self.dim_ordering == 'tf':
            self.W_shape = (self.nb_row, self.nb_col, 1,
                            2*self.nb_filter_complex + self.nb_filter_simple)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        self.W = self.add_weight(self.W_shape,
                                 initializer=functools.partial(self.init,
                                                               dim_ordering=self.dim_ordering),
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)

        if self.bias:
            self.b = self.add_weight((self.nb_filter_complex + self.nb_filter_simple,),
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

    def get_output_shape_for(self, input_shape):
        if self.dim_ordering == 'th':
            stack = input_shape[1]
            row = input_shape[2]
            col = input_shape[3]
        elif self.dim_ordering == 'tf':
            row = input_shape[1]
            col = input_shape[2]
            stack = input_shape[3]
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        row_out = conv_output_length(row, self.nb_row,
                                     self.border_mode, self.subsample[0])
        col_out = conv_output_length(col, self.nb_col,
                                     self.border_mode, self.subsample[1])

        if self.dim_ordering == 'th':
            return (input_shape[0], (self.nb_filter_complex + self.nb_filter_simple)*stack, row_out, col_out)
        elif self.dim_ordering == 'tf':
            return (input_shape[0], row_out, col_out, stack*(self.nb_filter_complex + self.nb_filter_simple))
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        output_shape = [-1] + list(self.get_output_shape_for(input_shape)[1:])
        xshape = K.shape(x)
        if self.dim_ordering == 'th':
            x = K.reshape(x, (-1, 1, xshape[2], xshape[3]))
        elif self.dim_ordering == 'tf':
            x = K.permute_dimensions(x, (0, 3, 1, 2))
            x = K.reshape(x, (-1, xshape[1], xshape[2], 1))

        conv_out = K.conv2d(x, self.W, strides=self.subsample,
                            border_mode=self.border_mode,
                            dim_ordering=self.dim_ordering,
                            filter_shape=self.W_shape)

        if self.dim_ordering == 'th':
            # Complex-cell filter operation
            conv_out1 = K.sqrt(K.square(conv_out[:, :self.nb_filter_complex, :, :]) + K.square(conv_out[:, self.nb_filter_complex:2*self.nb_filter_complex, :, :]) + K.epsilon())
            # Simple-cell filter operation
            conv_out2 = K.concatenate([conv_out1, conv_out[:, 2*self.nb_filter_complex:, :, :]], axis=1)
        elif self.dim_ordering == 'tf':
            # Complex-cell filter operation
            conv_out1 = K.sqrt(K.square(conv_out[:, :, :, :self.nb_filter_complex]) + K.square(conv_out[:, :, :, self.nb_filter_complex:2*self.nb_filter_complex]) + K.epsilon())
            # Simple-cell filter operation
            conv_out2 = K.concatenate([conv_out1, conv_out[:, :, :, 2*self.nb_filter_complex:]], axis=3)

        if self.bias:
            if self.dim_ordering == 'th':
                conv_out2 += K.reshape(self.b, (1, self.nb_filter_complex + self.nb_filter_simple, 1, 1))
            elif self.dim_ordering == 'tf':
                conv_out2 += K.reshape(self.b, (1, 1, 1, self.nb_filter_complex + self.nb_filter_simple))
                conv_out2 = K.reshape(conv_out2, [-1, xshape[3], output_shape[1], output_shape[2], self.nb_filter_complex + self.nb_filter_simple])
                conv_out2 = K.permute_dimensions(conv_out2, (0, 2, 3, 1, 4))

        conv_out2 = self.activation(conv_out2)

        return K.reshape(conv_out2, output_shape)

    def get_config(self):
        config = {'nb_filter_simple': self.nb_filter_simple,
                  'nb_filter_complex': self.nb_filter_complex,
                  'nb_row': self.nb_row,
                  'nb_col': self.nb_col,
                  'dim_ordering': self.dim_ordering,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'border_mode': self.border_mode,
                  'subsample': self.subsample,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'bias': self.bias}
        base_config = super(Convolution2DEnergy_Scatter, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Convolution2DEnergy(Layer):
    def __init__(self, nb_filter_simple, nb_filter_complex, nb_row, nb_col,
                 init='glorot_uniform', activation='relu', weights=None,
                 border_mode='valid', subsample=(1, 1), dim_ordering=K.image_dim_ordering(),
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        if border_mode not in {'valid', 'same'}:
            raise Exception('Invalid border mode for Convolution2DEnergy:', border_mode)
        self.nb_filter_simple = nb_filter_simple
        self.nb_filter_complex = nb_filter_complex
        self.nb_row = nb_row
        self.nb_col = nb_col
        self.init = initializations.get(init, dim_ordering=dim_ordering)
        self.activation = activations.get(activation)
        assert border_mode in {'valid', 'same'}, 'border_mode must be in {valid, same}'
        self.border_mode = border_mode
        self.subsample = tuple(subsample)
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.UnitNormOrthogonal(nb_filter_complex, dim_ordering)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.input_spec = [InputSpec(ndim=4)]
        self.initial_weights = weights
        super(Convolution2DEnergy, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 4
        self.input_spec = [InputSpec(shape=input_shape)]

        if self.dim_ordering == 'th':
            stack_size = input_shape[1]
            self.W_shape = (2*self.nb_filter_complex + self.nb_filter_simple,
                            stack_size, self.nb_row, self.nb_col)
        elif self.dim_ordering == 'tf':
            stack_size = input_shape[3]
            self.W_shape = (self.nb_row, self.nb_col, stack_size,
                            2*self.nb_filter_complex + self.nb_filter_simple)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        self.W = self.add_weight(self.W_shape,
                                 initializer=functools.partial(self.init,
                                                               dim_ordering=self.dim_ordering),
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)

        if self.bias:
            self.b = self.add_weight((self.nb_filter_complex + self.nb_filter_simple,),
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

    def get_output_shape_for(self, input_shape):
        if self.dim_ordering == 'th':
            row = input_shape[2]
            col = input_shape[3]
        elif self.dim_ordering == 'tf':
            row = input_shape[1]
            col = input_shape[2]
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        row_out = conv_output_length(row, self.nb_row,
                                     self.border_mode, self.subsample[0])
        col_out = conv_output_length(col, self.nb_col,
                                     self.border_mode, self.subsample[1])

        if self.dim_ordering == 'th':
            return (input_shape[0], (self.nb_filter_complex + self.nb_filter_simple), row_out, col_out)
        elif self.dim_ordering == 'tf':
            return (input_shape[0], row_out, col_out, (self.nb_filter_complex + self.nb_filter_simple))
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def call(self, x, mask=None):
        conv_out = K.conv2d(x, self.W, strides=self.subsample,
                            border_mode=self.border_mode,
                            dim_ordering=self.dim_ordering,
                            filter_shape=self.W_shape)

        if self.dim_ordering == 'th':
            # Complex-cell filter operation
            conv_out1 = K.sqrt(K.square(conv_out[:, :self.nb_filter_complex, :, :]) + K.square(conv_out[:, self.nb_filter_complex:2*self.nb_filter_complex, :, :]) + K.epsilon())
            # Simple-cell filter operation
            conv_out2 = K.concatenate([conv_out1, conv_out[:, 2*self.nb_filter_complex:, :, :]], axis=1)
        elif self.dim_ordering == 'tf':
            # Complex-cell filter operation
            conv_out1 = K.sqrt(K.square(conv_out[:, :, :, :self.nb_filter_complex]) + K.square(conv_out[:, :, :, self.nb_filter_complex:2*self.nb_filter_complex]) + K.epsilon())
            # Simple-cell filter operation
            conv_out2 = K.concatenate([conv_out1, conv_out[:, :, :, 2*self.nb_filter_complex:]], axis=3)

        if self.bias:
            if self.dim_ordering == 'th':
                conv_out2 += K.reshape(self.b, (1, self.nb_filter_complex + self.nb_filter_simple, 1, 1))
            elif self.dim_ordering == 'tf':
                conv_out2 += K.reshape(self.b, (1, 1, 1, self.nb_filter_complex + self.nb_filter_simple))

        return self.activation(conv_out2)

    def get_config(self):
        config = {'nb_filter_simple': self.nb_filter_simple,
                  'nb_filter_complex': self.nb_filter_complex,
                  'nb_row': self.nb_row,
                  'nb_col': self.nb_col,
                  'dim_ordering': self.dim_ordering,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'border_mode': self.border_mode,
                  'subsample': self.subsample,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'bias': self.bias}
        base_config = super(Convolution2DEnergy, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
