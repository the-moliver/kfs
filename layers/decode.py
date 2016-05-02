# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division

import numpy as np

from keras import backend as K
from keras.engine import InputSpec, Layer
from keras.layers.convolutional import conv_output_length

# Differentiable Preprocessing for fMRI



class ImageOpt(Layer):
    def __init__(self):
        super(ImageOpt, self).__init__()
        self.input_spec = [InputSpec(ndim=3)]

    def call(self, x, mask=None):
        return x

    def get_config(self):
        config = {}
        base_config = super(ImageOpt, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class TemporalFilter(Layer):
    '''TemporalFilter for fMRI
    '''
    def __init__(self, real_filts=None, complex_filts=None,
                 border_mode='valid', subsample_length=1,
                 input_dim=None, input_length=None, **kwargs):

        if border_mode not in {'valid', 'same'}:
            raise Exception('Invalid border mode for Convolution1D:', border_mode)
        assert border_mode in {'valid', 'same'}, 'border_mode must be in {valid, same}'
        self.border_mode = border_mode
        self.subsample_length = subsample_length
        self.real_filts = real_filts
        self.complex_filts = complex_filts
        self.nb_filter = 0
        if real_filts is not None:
            self.nb_filter += real_filts.shape[0]
        if complex_filts is not None:
            self.nb_filter += complex_filts.shape[1]
        if real_filts is None:
            self.filter_length = complex_filts.shape[3]
        else:
            self.filter_length = real_filts.shape[2]
        self.subsample = (subsample_length, 1)

        self.input_spec = [InputSpec(ndim=3)]
        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(TemporalFilter, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_dim = input_shape[2]
        # self.W_shape = (self.nb_filter, input_dim, self.filter_length, 1)
        # self.W = self.init(self.W_shape, name='{}_W'.format(self.name))
        # self.b = K.zeros((self.nb_filter,), name='{}_b'.format(self.name))
        if self.real_filts is not None:
            self.W_r = K.variable(self.real_filts, name='{}_W_r'.format(self.name))
        if self.complex_filts is not None:
            self.W_c1 = K.variable(self.complex_filts[0], name='{}_W_c1'.format(self.name))
            self.W_c2 = K.variable(self.complex_filts[1], name='{}_W_c2'.format(self.name))

    def get_output_shape_for(self, input_shape):
        length = conv_output_length(input_shape[1],
                                    self.filter_length,
                                    self.border_mode,
                                    self.subsample[0])
        return (input_shape[0], length, self.nb_filter)

    def call(self, x, mask=None):
        x = K.permute_dimensions(x, (0, 2, 1))
        # x = K.squeeze(x, axis=0)
        x = K.reshape(x, (-1, self.input_length))
        x = K.expand_dims(x, 1)
        x = K.expand_dims(x, -1)
        if self.real_filts is not None:
            conv_out_r = K.conv2d(x, self.W_r, strides=self.subsample,
                                  border_mode=self.border_mode,
                                  dim_ordering='th')
        else:
            conv_out_r = x

        if self.complex_filts is not None:
            conv_out_c1 = K.conv2d(x, self.W_c1, strides=self.subsample,
                                   border_mode=self.border_mode,
                                   dim_ordering='th')
            conv_out_c2 = K.conv2d(x, self.W_c2, strides=self.subsample,
                                   border_mode=self.border_mode,
                                   dim_ordering='th')
            conv_out_c = K.sqrt(K.square(conv_out_c1) + K.square(conv_out_c2) + K.epsilon())
            output = K.concatenate((conv_out_r, conv_out_c), axis=1)
        else:
            output = conv_out_r

        output_shape = self.get_output_shape_for((None, self.input_length, self.input_dim))
        output = K.squeeze(output, 3)  # remove the dummy 3rd dimension
        output = K.permute_dimensions(output, (2, 1, 0))
        output = K.reshape(output, (-1, output_shape[1], output.shape[1]*output.shape[2]))
        return output

    def get_config(self):
        config = {'border_mode': self.border_mode,
                  'subsample_length': self.subsample_length,
                  'input_dim': self.input_dim,
                  'input_length': self.input_length}
        base_config = super(TemporalFilter, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class Rescale(Layer):
    '''Apply z-score scaling used in fits
    '''
    def __init__(self, means, stds, input_dim=None, **kwargs):
        self.input_dim = input_dim

        self.means = means
        self.stds = stds
        self.input_spec = [InputSpec(ndim=2)]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(Rescale, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.output_dim = input_dim
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, input_dim))]

        self.means = K.variable(self.means,
                                name='{}_means'.format(self.name))
        self.stds = K.variable(self.stds,
                               name='{}_stds'.format(self.name))

    def call(self, x, mask=None):
        return (x - self.means)/self.stds

    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], input_shape[1])

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'input_dim': self.input_dim}
        base_config = super(Rescale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class FixedDense(Layer):
    '''Fixed Weights from ridge, etc.
    '''
    def __init__(self, weights, biases, input_dim=None, **kwargs):

        self.input_dim = weights.shape[0]
        self.output_dim = weights.shape[1]

        self.weights = weights
        self.biases = biases

        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=2)]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(FixedDense, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, input_dim))]

        self.W = K.variable(self.weights,
                            name='{}_W'.format(self.name))
        self.b = K.variable(self.biases,
                            name='{}_b'.format(self.name))

    def call(self, x, mask=None):
        return K.dot(x, self.W) + self.b

    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'input_dim': self.input_dim}
        base_config = super(FixedDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
