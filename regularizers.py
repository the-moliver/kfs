from __future__ import absolute_import
from keras import backend as K
from keras.regularizers import Regularizer
import theano
import numpy as np

if K.backend() == 'theano':
    from theano import tensor as T


class XCovRegularizer(Regularizer):
    """Cross-Covariance Regularization for disentagling factors

    # Arguments
        gamma: Float; regularization factor.
        axis: Int; Axis along which to compute cross-covariance
        division_idx: Int; Slice where to divide matrix to compute
                           cross covariance between parts

    # References
        - [Discovering Hidden Factors of Variation in Deep Networks. Brian Cheung et al, 2015](https://arxiv.org/pdf/1412.6583.pdf)
    """
    def __init__(self, gamma=0., axis=1, division_idx=None):
        self.gamma = K.cast_to_floatx(gamma)
        self.axis = []
        self.axis.append(axis)
        self.division_idx = division_idx

    def set_layer(self, layer):
        self.layer = layer

    def __call__(self, x):
        regularization = 0
        dimorder = self.axis + list(set(range(K.ndim(x))) - set(self.axis))
        x = K.permute_dimensions(x, dimorder)
        x = x.reshape((x.shape[0], -1))
        x -= K.mean(x, axis=1, keepdims=True)
        if self.division_idx is not None:
            regularization += .5*K.sum(K.square(K.dot(x[:self.division_idx], x[self.division_idx:].T)/x.shape[1]))
        else:
            regularization += .5*K.sum(K.square(K.dot(x, x.T)/x.shape[1]))
        return regularization

    def get_config(self):
        return {"name": self.__class__.__name__,
                "gamma": float(self.gamma),
                "axis": self.axis,
                "division_idx": self.division_idx}


class StochasticWeightRegularizer(Regularizer):
    """Regularizer for enforcing weight matrix to be a stochastic
    matrix, i.e. sums to 1 along a specified axis. Useful for
    Archetypal Analysis type things.

    # Arguments
        l1: Float; L1 regularization factor.
        l2: Float; L2 regularization factor.
        axis: Int; Axis along which to compute Laplacian operator
    """
    def __init__(self, l1=0., l2=0., axis=0):
        self.l1 = K.cast_to_floatx(l1)
        self.l2 = K.cast_to_floatx(l2)
        self.axis = axis

    def __call__(self, x):
        regularization = 0
        if self.l1:
            regularization += self.l1 * K.sum(K.abs(K.sum(x, axis=self.axis) - 1.))
        if self.l2:
            regularization += self.l2 * K.sum(K.square(K.sum(x, axis=self.axis) - 1.))
        return regularization

    def get_config(self):
        return {"name": self.__class__.__name__,
                "l1": float(self.l1),
                "l2": float(self.l2),
                "axis": self.axis}


class LaplacianRegularizer(Regularizer):
    """Regularizer for L1 and L2 Norm of the Laplacian
    operator applied along a specific axis of the weight tensor.

    # Arguments
        l1: Float; L1 regularization factor.
        l2: Float; L2 regularization factor.
        axis: Int; Axis along which to compute Laplacian operator
    """
    def __init__(self, l1=0., l2=0., axis=0):
        self.l1 = K.cast_to_floatx(l1)
        self.l2 = K.cast_to_floatx(l2)
        self.axis = []
        self.axis.append(axis)

    def __call__(self, x):
        regularization = 0
        dimorder = self.axis + list(set(range(K.ndim(x))) - set(self.axis))
        lp = laplacian1d(K.permute_dimensions(x, dimorder))
        if self.l1:
            regularization += K.sum(self.l1 * K.abs(lp))
        if self.l2:
            regularization += K.sum(self.l2 * K.square(lp))
        return regularization

    def get_config(self):
        return {"name": self.__class__.__name__,
                "l1": float(self.l1),
                "l2": float(self.l2),
                "axis": self.axis}


class TVRegularizer(Regularizer):
    """Total-Variation (TV) and Total-Variation of the gradient (TV2)
    regularization applied along specific axes of the weight tensor.

    # Arguments
        l1: Float; L1 regularization factor.
        l2: Float; L2 regularization factor.
        axis: Int; Axis along which to compute Laplacian operator

    # References
        - [TV+TV2 Regularization with Nonconvex Sparseness-Inducing Penalty for Image Restoration, Chengwu Lu & Hua Huang, 2014](http://downloads.hindawi.com/journals/mpe/2014/790547.pdf)
    """
    def __init__(self, TV=0., TV2=0., axes=[0, 1]):

        self.TV = K.cast_to_floatx(TV)
        self.TV2 = K.cast_to_floatx(TV2)
        self.axes = list(axes)

    def __call__(self, x):
        regularization = 0
        dimorder = self.axes + list(set(range(K.ndim(x))) - set(self.axes))
        x = K.permute_dimensions(x, dimorder)

        if self.TV:
            regularization += K.sum(K.sqrt(K.square(diffr(x)) + K.square(diffc(x)) + K.epsilon()))
        if self.TV2:
            regularization += K.sum(K.sqrt(K.square(diffrr(x)) + K.square(diffcc(x)) + 2*K.square(diffrc(x)) + K.epsilon()))
        return regularization

    def get_config(self):
        return {"name": self.__class__.__name__,
                "TV": float(self.TV),
                "TV2": float(self.TV2),
                "axes": self.axes}


def laplacian1d(X):
    Xout = K.zeros(X.shape.eval())
    if K.backend() == 'theano':
        Xout = T.set_subtensor(Xout[1:-1], X[2:] + X[:-2])
        Xout = T.set_subtensor(Xout[0], X[1] + X[0])
        Xout = T.set_subtensor(Xout[-1], X[-1] + X[-2])
    elif K.backend() == 'tensorflow':
        Xout[1:-1].assign(X[2:] + X[:-2])
        Xout[0].assign(X[1] + X[0])
        Xout[-1].assign(X[-1] + X[-2])
    return Xout - 2*X


def diffc(X):
    Xout = K.zeros(X.shape.eval())
    if K.backend() == 'theano':
        Xout = T.set_subtensor(Xout[:, 1:-1], X[:, 2:] - X[:, :-2])
        Xout = T.set_subtensor(Xout[:, 0], X[:, 1] - X[:, 0])
        Xout = T.set_subtensor(Xout[:, -1], X[:, -1] - X[:, -2])
    elif K.backend() == 'tensorflow':
        Xout[:, 1:-1].assign(X[:, 2:] - X[:, :-2])
        Xout[:, 0].assign(X[:, 1] - X[:, 0])
        Xout[:, -1].assign(X[:, -1] - X[:, -2])
    return Xout/2


def diffr(X):
    Xout = K.zeros(X.shape.eval())
    if K.backend() == 'theano':
        Xout = T.set_subtensor(Xout[1:-1, :], X[2:, :] - X[:-2, :])
        Xout = T.set_subtensor(Xout[0, :], X[1, :] - X[0, :])
        Xout = T.set_subtensor(Xout[-1, :], X[-1, :] - X[-2, :])
    elif K.backend() == 'tensorflow':
        Xout[1:-1, :].assign(X[2:, :] - X[:-2, :])
        Xout[0, :].assign(X[1, :] - X[0, :])
        Xout[-1, :].assign(X[-1, :] - X[-2, :])
    return Xout/2


def diffcc(X):
    Xout = K.zeros(X.shape.eval())
    if K.backend() == 'theano':
        Xout = T.set_subtensor(Xout[:, 1:-1], X[:, 2:] + X[:, :-2])
        Xout = T.set_subtensor(Xout[:, 0], X[:, 1] + X[:, 0])
        Xout = T.set_subtensor(Xout[:, -1], X[:, -1] + X[:, -2])
    elif K.backend() == 'tensorflow':
        Xout[:, 1:-1].assign(X[:, 2:] + X[:, :-2])
        Xout[:, 0].assign(X[:, 1] + X[:, 0])
        Xout[:, -1].assign(X[:, -1] + X[:, -2])
    return Xout - 2*X


def diffrr(X):
    Xout = K.zeros(X.shape.eval())
    if K.backend() == 'theano':
        Xout = T.set_subtensor(Xout[1:-1, :], X[2:, :] + X[:-2, :])
        Xout = T.set_subtensor(Xout[0, :], X[1, :] + X[0, :])
        Xout = T.set_subtensor(Xout[-1, :], X[-1, :] + X[-2, :])
    elif K.backend() == 'tensorflow':
        Xout[1:-1, :].assign(X[2:, :] + X[:-2, :])
        Xout[0, :].assign(X[1, :] + X[0, :])
        Xout[-1, :].assign(X[-1, :] + X[-2, :])
    return Xout - 2*X


def diffrc(X):
    Xout1 = K.zeros(X.shape.eval())
    Xout2 = K.zeros(X.shape.eval())
    if K.backend() == 'theano':
        Xout1 = T.set_subtensor(Xout1[1:-1, 1:-1], X[2:, 2:]+X[:-2, :-2])
        Xout1 = T.set_subtensor(Xout1[0, 0], X[1, 1]+X[0, 0])
        Xout1 = T.set_subtensor(Xout1[0, 1:-1], X[1, 2:]+X[0, :-2])
        Xout1 = T.set_subtensor(Xout1[0, -1], X[1, -1]+X[0, -2])
        Xout1 = T.set_subtensor(Xout1[1:-1, -1], X[2:, -1]+X[:-2, -2])
        Xout1 = T.set_subtensor(Xout1[-1, -1], X[-1, -1]+X[-2, -2])
        Xout1 = T.set_subtensor(Xout1[-1, 1:-1], X[-1, 2:]+X[-2, :-2])
        Xout1 = T.set_subtensor(Xout1[-1, 0], X[-1, 1]+X[-2, 0])
        Xout1 = T.set_subtensor(Xout1[1:-1, 0], X[2:, 1]+X[:-2, 0])
        Xout2 = T.set_subtensor(Xout2[1:-1, 1:-1], X[:-2, 2:]+X[2:, :-2])
        Xout2 = T.set_subtensor(Xout2[0, 0], X[0, 1]+X[1, 0])
        Xout2 = T.set_subtensor(Xout2[0, 1:-1], X[0, 2:]+X[1, :-2])
        Xout2 = T.set_subtensor(Xout2[0, -1], X[0, -1]+X[1, -2])
        Xout2 = T.set_subtensor(Xout2[1:-1, -1], X[:-2, -1]+X[2:, -2])
        Xout2 = T.set_subtensor(Xout2[-1, -1], X[-2, -1]+X[-1, -2])
        Xout2 = T.set_subtensor(Xout2[-1, 1:-1], X[-2, 2:]+X[-1, :-2])
        Xout2 = T.set_subtensor(Xout2[-1, 0], X[-2, 1]+X[-1, 0])
        Xout2 = T.set_subtensor(Xout2[1:-1, 0], X[:-2, 1]+X[2:, 0])
    elif K.backend() == 'tensorflow':
        Xout1[1:-1, 1:-1].assign(X[2:, 2:]+X[:-2, :-2])
        Xout1[0, 0].assign(X[1, 1]+X[0, 0])
        Xout1[0, 1:-1].assign(X[1, 2:]+X[0, :-2])
        Xout1[0, -1].assign(X[1, -1]+X[0, -2])
        Xout1[1:-1, -1].assign(X[2:, -1]+X[:-2, -2])
        Xout1[-1, -1].assign(X[-1, -1]+X[-2, -2])
        Xout1[-1, 1:-1].assign(X[-1, 2:]+X[-2, :-2])
        Xout1[-1, 0].assign(X[-1, 1]+X[-2, 0])
        Xout1[1:-1, 0].assign(X[2:, 1]+X[:-2, 0])
        Xout2[1:-1, 1:-1].assign(X[:-2, 2:]+X[2:, :-2])
        Xout2[0, 0].assign(X[0, 1]+X[1, 0])
        Xout2[0, 1:-1].assign(X[0, 2:]+X[1, :-2])
        Xout2[0, -1].assign(X[0, -1]+X[1, -2])
        Xout2[1:-1, -1].assign(X[:-2, -1]+X[2:, -2])
        Xout2[-1, -1].assign(X[-2, -1]+X[-1, -2])
        Xout2[-1, 1:-1].assign(X[-2, 2:]+X[-1, :-2])
        Xout2[-1, 0].assign(X[-2, 1]+X[-1, 0])
        Xout2[1:-1, 0].assign(X[:-2, 1]+X[2:, 0])
    return (Xout1 - Xout2)/4


from keras.utils.generic_utils import get_from_module
def get(identifier, kwargs=None):
    return get_from_module(identifier, globals(), 'regularizer', instantiate=True, kwargs=kwargs)