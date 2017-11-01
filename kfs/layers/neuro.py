# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division

import numpy as np

from keras import backend as K
from keras import activations, initializers, regularizers
from kfs import constraints
from keras import constraints as kconstraints
from keras.engine import InputSpec, Layer
from keras.utils import conv_utils


if K.backend() == 'theano':
    from theano import tensor as T


def step_init(params):
    step = 2./(params[0]-1.)
    steps = params[1]*(np.arange(0, 2+step, step)[:params[0]] - 1.)[None, :, None]
    return steps


def step_init2(params):
    step = 2./(params[0]-1.)
    steps = params[1]*(np.arange(0, 2+step, step)[:params[0]] - 1.)
    return np.tile(steps, [params[2], 1])

class Convolution2DEnergy_TemporalBasis_GaussianRF(Layer):
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
                                padding='same',
                                input_shape=(3, 256, 256)))
        # now model.output_shape == (None, 4*7, 8+16, 256, 256)
    ```
    # Arguments
        filters_simple: Number of simple-cell filters to use.
        filters_complex: Number of complex-cell filters to use.
        spatial_kernel_size: Tuple containing number of rows and columns in the convolution kernel.
        filters_temporal: Number of temporal amplitude filters
        temporal_frequencies: Number of temporal frequencies (odd number)
        temporal_frequencies_initial_max: Maximum temporal frequency, temporal frequencies initialized
            as (-tf_max..., 0, ..., tf_max)
        spatial_kernel_initializer: name of initialization function for the spatial kernel weights
            (see [initializers](../initializers.md))
        temporal_kernel_initializer: name of initialization function for the temporal kernel weights
            (see [initializers](../initializers.md))
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list of numpy arrays to set as initial weights.
        padding: 'valid', 'same' or 'full'
            ('full' requires the Theano backend).
        strides: tuple of length 2. Factor by which to strides output.
            Also called strides elsewhere.
        spatial_kernel_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the spatial kernel weights matrix.
        temporal_kernel_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L2 or Laplacian regularization), applied to the temporal amplitude
            weights matrix.
        bias_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        spatial_kernel_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        temporal_kernel_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        bias_constraint: instance of the [constraints](../constraints.md) module,
            applied to the bias.
        data_format: 'channels_first' or 'channels_last'. In 'channels_first' mode, the channels dimension
            (the depth) is at index 1, in 'channels_last' mode is it at index 3.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
        use_bias: whether to include a bias
            (i.e. make the layer affine rather than linear).
    # Input shape
        5D tensor with shape:
        `(samples, time_steps, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, time_steps, rows, cols, channels)` if data_format='channels_last'.
    # Output shape
        5D tensor with shape:
        `(samples, nb_temporal_filter, nb_filter, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, nb_temporal_filter, new_rows, new_cols, nb_filter)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to padding.
    """

    def __init__(self, neurons,
                 filters_simple,
                 filters_complex,
                 filters_temporal,
                 spatial_kernel_size,
                 temporal_frequencies,
                 spatial_kernel_initializer='glorot_uniform',
                 temporal_kernel_initializer='glorot_uniform',
                 temporal_frequencies_initializer=step_init,
                 temporal_frequencies_initial_max=2,
                 temporal_frequencies_scaling=10,
                 bias_initializer='zeros',
                 activation='relu',
                 padding='valid',
                 strides=(1, 1),
                 dilation_rate=(1, 1),
                 data_format=K.image_data_format(),
                 spatial_kernel_regularizer=None,
                 temporal_kernel_regularizer=None,
                 temporal_frequencies_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 spatial_kernel_constraint=None,
                 temporal_kernel_constraint=None,
                 temporal_frequencies_constraint=None,
                 bias_constraint=None,
                 use_bias=True,
                 centers_initializer='zeros',
                 centers_regularizer=None,
                 centers_constraint=None,
                 stds_initializer='ones',
                 stds_regularizer=None,
                 stds_constraint=None,
                 gauss_scale=100, **kwargs):

        self.neurons = neurons
        self.gauss_scale = gauss_scale
        self.centers_initializer = initializers.get(centers_initializer)
        self.stds_initializer = initializers.get(stds_initializer)
        self.centers_regularizer = regularizers.get(centers_regularizer)
        self.stds_regularizer = regularizers.get(stds_regularizer)
        self.centers_constraint = constraints.get(centers_constraint)
        self.stds_constraint = constraints.get(stds_constraint)

        self.filters_simple = filters_simple
        self.filters_complex = filters_complex
        self.filters_temporal = filters_temporal
        self.spatial_kernel_size = spatial_kernel_size
        self.temporal_frequencies = temporal_frequencies
        self.temporal_frequencies_initial_max = np.float32(temporal_frequencies_initial_max)
        self.temporal_frequencies_scaling = np.float32(temporal_frequencies_scaling)
        self.spatial_kernel_initializer = initializers.get(spatial_kernel_initializer)
        self.temporal_kernel_initializer = initializers.get(temporal_kernel_initializer)
        self.temporal_frequencies_initializer = initializers.get(temporal_frequencies_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.activation = activations.get(activation)
        assert padding in {'valid', 'same'}, 'padding must be in {valid, same}'
        self.padding = padding
        self.strides = strides
        self.dilation_rate = dilation_rate
        assert data_format in {'channels_last', 'channels_first'}, 'data_format must be in {channels_first, channels_last}'
        self.data_format = data_format

        self.spatial_kernel_regularizer = regularizers.get(spatial_kernel_regularizer)
        self.temporal_kernel_regularizer = regularizers.get(temporal_kernel_regularizer)
        self.temporal_frequencies_regularizer = regularizers.get(temporal_frequencies_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.spatial_kernel_constraint = constraints.UnitNormOrthogonal(self.filters_complex + self.filters_simple)
        self.temporal_kernel_constraint = constraints.get(temporal_kernel_constraint)
        self.temporal_frequencies_constraint = constraints.get(temporal_frequencies_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.epsilon = K.epsilon()

        self.use_bias = use_bias
        self.input_spec = [InputSpec(ndim=5)]

        super(Convolution2DEnergy_TemporalBasis_GaussianRF, self).__init__(**kwargs)

    def build(self, input_shape):
        # assert len(input_shape) == 5
        if self.data_format == 'channels_first':
            channel_axis = 2
        else:
            channel_axis = -1
        # if input_shape[channel_axis] is None:
        #     raise ValueError('The channel dimension of the inputs '
        #                      'should be defined. Found `None`.')

        self.delays = input_shape[0][1]
        input_dim = input_shape[0][channel_axis]
        spatial_kernel_shape = self.spatial_kernel_size + (input_dim, 2*self.filters_complex + 2*self.filters_simple)

        self.spatial_kernel = self.add_weight(spatial_kernel_shape,
                                              initializer=self.spatial_kernel_initializer,
                                              name='spatial_kernel',
                                              regularizer=self.spatial_kernel_regularizer,
                                              constraint=self.spatial_kernel_constraint)

        self.temporal_kernel = K.pattern_broadcast(self.add_weight((self.delays, 1, self.filters_temporal),
                                               initializer=self.temporal_kernel_initializer,
                                               name='temporal_kernel',
                                               regularizer=self.temporal_kernel_regularizer,
                                               constraint=self.temporal_kernel_constraint), [False, True, False])

        if self.use_bias:
            self.bias = self.add_weight((self.filters_complex + self.filters_simple,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.temporal_freqs = K.pattern_broadcast(self.add_weight((self.temporal_frequencies, self.temporal_frequencies_initial_max/self.temporal_frequencies_scaling),
                                              initializer=step_init,
                                              name='temporal_frequencies',
                                              regularizer=self.temporal_frequencies_regularizer,
                                              constraint=self.temporal_frequencies_constraint), [True, False, True])

        self.delays_pi = K.pattern_broadcast(K.constant(2 * np.pi * np.arange(0, 1 + 1. / (self.delays - 1), 1. / (self.delays - 1))[:self.delays][:, None, None], name='delays_pi'), [False, True, True])

        self.WT = K.zeros((4*self.delays, 3*self.filters_temporal*self.temporal_frequencies))


        # kernel_shape = [1] * (ndim) + [self.neurons]
        # kernel_broadcast = [True] * (ndim) + [False]

        # self.filter_axes = np.arange(ndim - 2, ndim)

        # for i in self.filter_axes:
        #     kernel_shape[i] = input_shape[0][i]
        #     kernel_broadcast[i] = False
        # kernel_shape[0] = -1
        # kernel_broadcast[0] = False


        self.centers = self.add_weight((2, self.neurons),
                                      initializer=self.centers_initializer,
                                      name='centers',
                                      regularizer=self.centers_regularizer,
                                      constraint=self.centers_constraint)

        self.stds = self.add_weight((self.neurons,),
                                      initializer=self.stds_initializer,
                                      name='stds',
                                      regularizer=self.stds_regularizer,
                                      constraint=self.stds_constraint)


        if self.data_format == 'channels_first':
            conv_dim2 = input_shape[0][3]
            conv_dim3 = input_shape[0][4]
        elif self.data_format == 'channels_last':
            conv_dim2 = input_shape[0][2]
            conv_dim3 = input_shape[0][3]
        else:
            raise Exception('Invalid data_format: ' + self.data_format)

        conv_dim2 = conv_utils.conv_output_length(conv_dim2,
                                                  self.spatial_kernel_size[0],
                                                  padding=self.padding,
                                                  stride=self.strides[0],
                                                  dilation=self.dilation_rate[0])
        conv_dim3 = conv_utils.conv_output_length(conv_dim3,
                                                  self.spatial_kernel_size[1],
                                                  padding=self.padding,
                                                  stride=self.strides[1],
                                                  dilation=self.dilation_rate[1])


        maxshape = np.max([conv_dim2, conv_dim3])
        dx = 1. / (maxshape - 1)
        self.dx = 2*dx
        maxrange = 2*(np.arange(0, 1+dx, dx) - .5)

        X = maxrange[np.int((maxshape - conv_dim3) / 2): np.int((maxshape + conv_dim3) / 2)]    
        Y = maxrange[np.int((maxshape - conv_dim2) / 2): np.int((maxshape + conv_dim2) / 2)]

        (XX, YY) = np.meshgrid(X, Y)

        self.XX = K.variable(XX)
        self.YY = K.variable(YY)

        # self.kernel_broadcast = kernel_broadcast
        # self.kernel_shape = kernel_shape


        # Set input spec.
        # self.input_spec = InputSpec(ndim=5,
        #                             axes={channel_axis: input_dim})
        self.built = True

    def compute_output_shape(self, input_shape):

        if self.data_format == 'channels_first':
            return (input_shape[0][0], self.filters_temporal*self.temporal_frequencies, (self.filters_complex + self.filters_simple), self.neurons)
        elif self.data_format == 'channels_last':
            return (input_shape[0][0], self.filters_temporal*self.temporal_frequencies, self.neurons, (self.filters_complex + self.filters_simple))
        else:
            raise Exception('Invalid data_format: ' + self.data_format)

    def call(self, inputs):

        stim = inputs[0]
        center = inputs[1]


        xshape = K.shape(stim)
        stim = K.reshape(stim, (-1, xshape[2], xshape[3], xshape[4]))

        tfs = self.temporal_frequencies_scaling*self.delays_pi*self.temporal_freqs

        sin_step = K.reshape(K.sin(tfs)*self.temporal_kernel, (-1, self.filters_temporal*self.temporal_frequencies))
        cos_step = K.reshape(K.cos(tfs)*self.temporal_kernel, (-1, self.filters_temporal*self.temporal_frequencies))

        w0t = K.concatenate((cos_step, -sin_step), axis=0)
        w1t = K.concatenate((sin_step, cos_step), axis=0)
        wt = K.concatenate((w0t, w1t), axis=1)

        self.WT = T.set_subtensor(self.WT[:2*self.delays, :2*self.filters_temporal*self.temporal_frequencies], wt)
        self.WT = T.set_subtensor(self.WT[2*self.delays:, 2*self.filters_temporal*self.temporal_frequencies:], w0t)

        conv_out1 = K.conv2d(
            stim,
            self.spatial_kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)

        conv_out1_shape = K.shape(conv_out1)

        conv_out1 = K.reshape(conv_out1, (-1, self.delays, conv_out1_shape[1], conv_out1_shape[2], conv_out1_shape[3]))


        centers_x = self.XX[None, None, None, :, :, None] - center[:, :, 0, None, None, None, None] - self.centers[0][None, None, None, None, None, :]
        centers_y = self.YY[None, None, None, :, :, None] - center[:, :, 1, None, None, None, None] - self.centers[1][None, None, None, None, None, :]
        senv = self.stds[None, None, None, None, None, :]
        gauss = self.gauss_scale * (K.square(self.dx) / (2 * np.pi * K.square(senv) + K.epsilon()))*K.exp(-(K.square(centers_x) + K.square(centers_y))/(2.0 * K.square(senv)))

        if K.backend() == 'theano':
            conv_out1 = K.sum(conv_out1[..., None] * K.pattern_broadcast(gauss, [False, False, True, False, False, False]), axis=[3, 4], keepdims=False)

        if self.data_format == 'channels_first':
            # samps x delays x stack x neurons
            conv_out1 = K.permute_dimensions(conv_out1, (0, 2, 3, 1))
            # samps x stack x neurons x delays
        elif self.data_format == 'channels_last':
            # samps x delays x neurons x stack
            conv_out1 = K.permute_dimensions(conv_out1, (0, 3, 2, 1))
            # samps x stack x neurons x delays
        else:
            raise Exception('Invalid data_format: ' + self.data_format)

        # split out complex and simple filter pairs
        conv_out1234 = K.concatenate([conv_out1[:, :self.filters_complex, :, :], conv_out1[:, self.filters_complex+self.filters_simple:2*self.filters_complex+self.filters_simple, :, :], conv_out1[:, self.filters_complex:self.filters_complex + self.filters_simple, :, :], conv_out1[:, 2*self.filters_complex + self.filters_simple:, :, :]], axis=3)

        # apply temporal trade-off to get temporal filter outputs and compute complex and simple outputs
        conv_out = K.dot(conv_out1234, self.WT)

        conv_outlin = conv_out[..., 2*self.filters_temporal*self.temporal_frequencies:]

        conv_out = K.square(conv_out[..., :2*self.filters_temporal*self.temporal_frequencies])
        conv_out = K.sqrt(conv_out[..., :self.filters_temporal*self.temporal_frequencies] + conv_out[..., self.filters_temporal*self.temporal_frequencies:] + self.epsilon)

        output = K.concatenate([conv_out, conv_outlin], axis=1)

        if self.data_format == 'channels_first':
            output = K.permute_dimensions(output, (0, 3, 1, 2))
            # samps x temporal_filters*temporal_frequencies x stack x neurons
        elif self.data_format == 'channels_last':
            output = K.permute_dimensions(output, (0, 3, 2, 1))
            # samps x temporal_filters*temporal_frequencies x neurons x stack

        if self.bias:
            if self.data_format == 'channels_first':
                output += K.reshape(self.bias, (1, 1, self.filters_complex + self.filters_simple, 1))
            elif self.data_format == 'channels_last':
                output += K.reshape(self.bias, (1, 1, 1, self.filters_complex + self.filters_simple))

        output = self.activation(output)
        return output

    def get_config(self):
        config = {'filters_simple': self.filters_simple,
                  'filters_complex': self.filters_complex,
                  'filters_temporal': self.filters_temporal,
                  'spatial_kernel_size': self.spatial_kernel_size,
                  'temporal_frequencies': self.temporal_frequencies,
                  'temporal_frequencies_initial_max': self.temporal_frequencies_initial_max,
                  'temporal_frequencies_scaling': self.temporal_frequencies_scaling,
                  'strides': self.strides,
                  'padding': self.padding,
                  'data_format': self.data_format,
                  'dilation_rate': self.dilation_rate,
                  'activation': activations.serialize(self.activation),
                  'use_bias': self.use_bias,
                  'spatial_kernel_initializer': initializers.serialize(self.spatial_kernel_initializer),
                  'temporal_kernel_initializer': initializers.serialize(self.temporal_kernel_initializer),
                  'temporal_frequencies_initializer': initializers.serialize(self.temporal_frequencies_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'spatial_kernel_regularizer': regularizers.serialize(self.spatial_kernel_regularizer),
                  'temporal_kernel_regularizer': regularizers.serialize(self.temporal_kernel_regularizer),
                  'temporal_frequencies_regularizer': regularizers.serialize(self.temporal_frequencies_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'spatial_kernel_constraint': constraints.serialize(self.spatial_kernel_constraint),
                  'temporal_kernel_constraint': constraints.serialize(self.temporal_kernel_constraint),
                  'temporal_frequencies_constraint': constraints.serialize(self.temporal_frequencies_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'neurons': self.neurons,
                  'gauss_scale': self.gauss_scale,
                  'centers_initializer': initializers.serialize(self.centers_initializer),
                  'stds_initializer': initializers.serialize(self.stds_initializer),
                  'centers_regularizer': regularizers.serialize(self.centers_regularizer),
                  'stds_regularizer': regularizers.serialize(self.stds_regularizer),
                  'centers_constraint': constraints.serialize(self.centers_constraint),
                  'stds_constraint': constraints.serialize(self.stds_constraint),
                  }
        base_config = super(Convolution2DEnergy_TemporalBasis_GaussianRF, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GaussianReceptiveFields(Layer):

    def __init__(self, filters,
                 centers_initializer='zeros',
                 centers_regularizer=None,
                 centers_constraint=None,
                 stds_initializer='ones',
                 stds_regularizer=None,
                 stds_constraint=None,
                 gauss_scale=100,
                 **kwargs):
        self.filters = filters
        self.gauss_scale = gauss_scale
        super(GaussianReceptiveFields, self).__init__(**kwargs)
        self.centers_initializer = initializers.get(centers_initializer)
        self.stds_initializer = initializers.get(stds_initializer)
        self.centers_regularizer = regularizers.get(centers_regularizer)
        self.stds_regularizer = regularizers.get(stds_regularizer)
        self.centers_constraint = constraints.get(centers_constraint)
        self.stds_constraint = constraints.get(stds_constraint)

    def build(self, input_shape):
        ndim = len(input_shape[0])
        assert ndim >= 2

        kernel_shape = [1] * (ndim) + [self.filters]
        kernel_broadcast = [True] * (ndim) + [False]

        self.filter_axes = np.arange(ndim - 2, ndim)

        for i in self.filter_axes:
            kernel_shape[i] = input_shape[0][i]
            kernel_broadcast[i] = False
        kernel_shape[0] = -1
        kernel_broadcast[0] = False


        self.centers = self.add_weight((2, self.filters),
                                      initializer=self.centers_initializer,
                                      name='centers',
                                      regularizer=self.centers_regularizer,
                                      constraint=self.centers_constraint)

        self.stds = self.add_weight((self.filters,),
                                      initializer=self.stds_initializer,
                                      name='stds',
                                      regularizer=self.stds_regularizer,
                                      constraint=self.stds_constraint)


        maxshape = np.max(input_shape[0][-2:])
        dx = 1. / (maxshape - 1)
        self.dx = 2*dx
        maxrange = 2*(np.arange(0, 1+dx, dx) - .5)

        X = maxrange[np.int((maxshape - input_shape[0][-1]) / 2): np.int((maxshape + input_shape[0][-1]) / 2)]    
        Y = maxrange[np.int((maxshape - input_shape[0][-2]) / 2): np.int((maxshape + input_shape[0][-2]) / 2)]

        (XX, YY) = np.meshgrid(X, Y)

        self.XX = K.variable(XX)
        self.YY = K.variable(YY)

        self.kernel_broadcast = kernel_broadcast
        self.kernel_shape = kernel_shape
        self.built = True

    def call(self, inputs):
        stim = inputs[0]
        center = inputs[1]
        centers_x = self.XX[None, :, :, None] - center[:, 0, None, None, None] - self.centers[0][None, None, None, :]
        centers_y = self.YY[None, :, :, None] - center[:, 1, None, None, None] - self.centers[1][None, None, None, :]
        senv = self.stds[None, None, None, :]
        gauss = self.gauss_scale * (K.square(self.dx) / (2 * np.pi * K.square(senv) + K.epsilon()))*K.exp(-(K.square(centers_x) + K.square(centers_y))/(2.0 * K.square(senv)))
        # gauss = (1 / K.sqrt(2 * np.pi * K.square(senv) + K.epsilon()))*K.exp(-(K.square(centers_x) + K.square(centers_y))/(2.0 * K.square(senv)))
        # gauss /= K.max(gauss, axis=(1, 2), keepdims=True)
        gauss = K.reshape(gauss, self.kernel_shape)

        if K.backend() == 'theano':
            output = K.sum(stim[..., None] * K.pattern_broadcast(gauss, self.kernel_broadcast), axis=self.filter_axes, keepdims=False)
        else:
            output = K.sum(stim[..., None] * gauss, axis=self.filter_axes, keepdims=False)

        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0][:-2] + tuple([self.filters])


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
