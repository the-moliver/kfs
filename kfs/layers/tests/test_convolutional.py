import keras.backend as K
import numpy as np
from kfs.layers.convolutional import (Convolution2DEnergy_TemporalBasis,
                                    Convolution2DEnergy_TemporalCorrelation)

def _test_smoke(channel_order=None):

    from kfs.layers.convolutional import Convolution2DEnergy_TemporalBasis
    from keras.models import Sequential
    #from keras.layers import Flatten, Dense
    input_shape = (12, 3, 64, 64)
    if channel_order is None:
        channel_order = K.image_data_format()
    if channel_order == 'channels_last':
        input_shape = (12, 64, 64, 3)

    rng = np.random.RandomState(42)
    datums = rng.randn(6, 12, 3, 64, 64).astype('float32')
    if channel_order == 'channels_last':
        datums = datums.transpose(0, 1, 3, 4, 2)

    nn2 = Sequential()
    nn2.add(Convolution2DEnergy_TemporalCorrelation(8, 16, 4, (5, 5), 7,
                                            padding='same',
                                            temporal_kernel_size=5,
                                            input_shape=input_shape))
    nn2.compile(loss='mse', optimizer='sgd')

    pred2 = nn2.predict(datums)

    return nn2, nn2.predict(datums)


def test_smoke_channels_first():
    K.set_image_data_format('channels_first')
    _test_smoke('channels_first')


def test_smoke_channels_last():
    K.set_image_data_format('channels_last')
    _test_smoke('channels_last')


def _test_equivalence(channel_order=None):

    from kfs.layers.convolutional import Convolution2DEnergy_TemporalBasis
    from keras.models import Sequential
    #from keras.layers import Flatten, Dense
    input_shape = (12, 3, 64, 64)
    if channel_order is None:
        channel_order = K.image_data_format()
    if channel_order == 'channels_last':
        input_shape = (12, 64, 64, 3)


    nn = Sequential()
    nn.add(Convolution2DEnergy_TemporalBasis(8, 16, 4, (5, 5), 7,
                                            padding='same',
                                            input_shape=input_shape,
                                            data_format=channel_order))

    rng = np.random.RandomState(42)
    datums = rng.randn(6, 12, 3, 64, 64).astype('float32')
    if channel_order == 'channels_last':
        datums = datums.transpose(0, 1, 3, 4, 2)


    nn.compile(loss='mse', optimizer='sgd')

    nn2 = Sequential()
    nn2.add(Convolution2DEnergy_TemporalCorrelation(8, 16, 4, (5, 5), 7,
                                            padding='same',
                                            input_shape=input_shape,
                                            data_format=channel_order))
    nn2.compile(loss='mse', optimizer='sgd')
    nn2.set_weights(nn.get_weights())

    pred1 = nn.predict(datums)
    pred2 = nn2.predict(datums)
    assert ((pred1 - pred2) == 0.).all()

    return nn, nn.predict(datums), nn2, nn2.predict(datums)


def test_equivalence_channels_first():
    K.set_image_data_format('channels_first')
    _test_equivalence('channels_first')


def test_equivalence_channels_last():
    K.set_image_data_format('channels_last')
    _test_equivalence('channels_last')



