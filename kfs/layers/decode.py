# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division

import numpy as np

from keras import backend as K
from keras.engine import InputSpec, Layer
from keras import initializers
from keras import regularizers
from keras import constraints
# from keras.layers.convolutional import conv_output_length

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


class SpatioTemporalFilterSimple(Layer):
    '''TemporalFilter for fMRI
    '''
    def __init__(self, nb_simple, filter_delays, kernel_initializer='glorot_uniform', kernel_regularizer=None, kernel_constraint=None, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        self.nb_simple = nb_simple
        self.filter_delays = filter_delays
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)

        self.input_spec = [InputSpec(ndim=3)]
        super(SpatioTemporalFilterSimple, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(self.filter_delays, 1, input_shape[-1], self.nb_simple),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.input_spec = InputSpec(ndim=3)
        self.built = True

    def compute_output_shape(self, input_shape):
        length = input_shape[1] - self.filter_delays + 1
        return (input_shape[0], length, self.nb_simple)

    def call(self, x, mask=None):
        x = K.permute_dimensions(x, (0, 2, 1))
        x = K.expand_dims(x, -1)

        output = K.permute_dimensions(K.squeeze(K.conv2d(x, self.kernel), -1), (0, 2, 1))

        return output

    def get_config(self):
        config = {'nb_simple': self.nb_simple,
                  'filter_delays':self.filter_delays,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint)}
        base_config = super(SpatioTemporalFilterSimple, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SpatioTemporalFilterComplex(Layer):
    '''TemporalFilter for fMRI
    '''
    def __init__(self, nb_complex, filter_delays, kernel_initializer='glorot_uniform', kernel_regularizer=None, kernel_constraint=None, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        self.nb_complex = nb_complex
        self.filter_delays = filter_delays
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)

        self.input_spec = [InputSpec(ndim=3)]
        super(SpatioTemporalFilterComplex, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(self.filter_delays, 1, input_shape[-1], self.nb_complex),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.input_spec = InputSpec(ndim=3)
        self.built = True

    def compute_output_shape(self, input_shape):
        length = input_shape[1] - self.filter_delays + 1
        return (input_shape[0], length, self.nb_complex)

    def call(self, x, mask=None):
        x = K.permute_dimensions(x, (0, 2, 1))
        x = K.expand_dims(x, -1)

        output = K.square(K.permute_dimensions(K.squeeze(K.conv2d(x, self.kernel), -1), (0, 2, 1)))

        return output

    def get_config(self):
        config = {'nb_complex': self.nb_complex,
                  'filter_delays':self.filter_delays,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint)}
        base_config = super(SpatioTemporalFilterComplex, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SpatioTemporalFilter(Layer):
    '''TemporalFilter for fMRI
    '''
    def __init__(self, nb_simple, nb_complex, filter_delays, kernel_initializer='glorot_uniform', kernel_regularizer=None, kernel_constraint=None, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        self.nb_simple = nb_simple
        self.nb_complex = nb_complex
        self.filter_delays = filter_delays
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)

        self.input_spec = [InputSpec(ndim=3)]
        super(SpatioTemporalFilter, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(self.filter_delays, 1, input_shape[-1], self.nb_complex + self.nb_simple),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.input_spec = InputSpec(ndim=3)
        self.built = True

    def compute_output_shape(self, input_shape):
        length = input_shape[1] - self.filter_delays + 1
        return (input_shape[0], length, self.nb_simple + self.nb_complex)

    def call(self, x, mask=None):
        x = K.permute_dimensions(x, (0, 2, 1))
        x = K.expand_dims(x, -1)

        conv_out = K.permute_dimensions(K.squeeze(K.conv2d(x, self.kernel), -1), (0, 2, 1))

        conv_out_s = conv_out[:,:,:self.nb_simple]

        conv_out_c = K.square(conv_out[:,:,self.nb_simple:])

        output = K.concatenate((conv_out_s, conv_out_c), axis=-1)

        return output

    def get_config(self):
        config = {'nb_simple': self.nb_simple,
                  'nb_complex': self.nb_complex,
                  'filter_delays':self.filter_delays,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint)}
        base_config = super(SpatioTemporalFilter, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Rescale(Layer):
    '''Apply z-score scaling used in fits
    '''
    def __init__(self, means, stds, input_dim=None, **kwargs):
        self.input_dim = input_dim

        self.means = means
        self.stds = stds
        self.input_spec = [InputSpec(min_ndim=2)]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(Rescale, self).__init__(**kwargs)

    def build(self, input_shape):
        self.means = K.variable(self.means,
                                name='{}_means'.format(self.name))
        self.stds = K.variable(self.stds,
                               name='{}_stds'.format(self.name))
        self.built = True

    def call(self, x, mask=None):
        return (x - self.means)/self.stds

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'means': self.means,
                  'stds': self.stds}
        base_config = super(Rescale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))