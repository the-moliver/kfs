from __future__ import absolute_import
from keras import backend as K
from keras.regularizers import Regularizer
import theano
import numpy as np
from theano import tensor as T

class XCovWeightRegularizer(Regularizer):
    def __init__(self, l=1., axis=0):
        self.l = l
        self.axis = axis

    def set_param(self, p):
        self.p = p

    def __call__(self, loss):
        p = self.p
        if self.axis == 1:
            p = p.T

        # p -= K.mean(p, axis=0)
        pen = .5*K.sum(K.square(K.dot(p.T, p)/p.shape[0]))
        loss += pen * self.l
        return loss

    def get_config(self):
        return {"name": self.__class__.__name__,
                "l": self.l,
                "axis": self.axis}


class OrthogonalWeightRegularizer(Regularizer):
    def __init__(self, l=1.):
        self.l = l

    def set_param(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    def __call__(self, loss):
        loss += K.sum(K.sum(self.p1 * self.p2, axis=-1)**2) * self.l
        return loss

    def get_config(self):
        return {"name": self.__class__.__name__,
                "l": self.l}


class OrthogonalActivityRegularizer(Regularizer):
    def __init__(self, gamma=0.):
        self.gamma = gamma

    def set_layer(self, layer):
        self.layer = layer

    def __call__(self, loss):
        output = self.layer.get_output(True)
        output -= K.mean(output, axis=0)
        pen = K.sum(K.square(K.dot(output.T, output)/output.shape[0]))
        loss += self.gamma*pen
        return loss

    def get_config(self):
        return {"name": self.__class__.__name__,
                "gamma": self.gamma}


class XCovActivityRegularizer(Regularizer):
    def __init__(self, gamma=0., dividx=1):
        self.gamma = gamma
        self.dividx = dividx

    def set_layer(self, layer):
        self.layer = layer

    def __call__(self, loss):
        output = self.layer.get_output(True)
        output -= K.mean(output, axis=0)
        pen = .5*K.sum(K.square(K.dot(output[:, :self.dividx].T, output[:, self.dividx:])/output.shape[0]))
        loss += self.gamma*pen
        return loss

    def get_config(self):
        return {"name": self.__class__.__name__,
                "gamma": self.gamma}


class LaplacianWeightRegularizer(Regularizer):
    def __init__(self, l1=0., l2=0.):
        self.l1 = l1
        self.l2 = l2
        self.uses_learning_phase = True

    def set_param(self, p):
        self.p = p

    def __call__(self, loss):
        regularized_loss = loss + K.sum(abs(laplacian1d(self.p))) * self.l1
        regularized_loss += K.sum(laplacian1d(self.p) ** 2) * self.l2
        return K.in_train_phase(regularized_loss, loss)

    def get_config(self):
        return {"name": self.__class__.__name__,
                "l1": self.l1,
                "l2": self.l2}


class LaplacianRegularizer(Regularizer):
    def __init__(self, l1=0., l2=0., axis=0):
        self.l1 = l1
        self.l2 = l2
        self.axis = []
        self.axis.append(axis)
        self.uses_learning_phase = True

    def set_param(self, p):
        self.p = p

    def __call__(self, loss):
        dimorder = self.axis + list(set(range(K.ndim(self.p))) - set(self.axis))
        p = K.permute_dimensions(self.p, dimorder)
        lp = laplacian1d(p)
        regularized_loss = loss + K.sum(abs(lp)) * self.l1
        regularized_loss += K.sum(lp ** 2) * self.l2
        return K.in_train_phase(regularized_loss, loss)

    def get_config(self):
        return {"name": self.__class__.__name__,
                "l1": self.l1,
                "l2": self.l2}


class TVWeightRegularizer(Regularizer):
    def __init__(self, TV=0., TV2=0., rows=1, columns=1, colors=1):
        self.TV = TV
        self.TV2 = TV2
        self.rows = rows
        self.columns = columns
        self.colors = colors
        self.uses_learning_phase = True

    def set_param(self, p):
        self.p = p

    def __call__(self, loss):
        p = K.reshape(self.p, (self.colors, self.rows, self.columns, self.p.shape[1]))

        pen1 = K.sum(K.sqrt(K.square(diffr(p[0])) + K.square(diffc(p[0])) + K.epsilon()), axis=(0, 1))
        pen2 = K.sum(K.sqrt(K.square(diffrr(p[0])) + K.square(diffcc(p[0])) + 2*K.square(diffrc(p[0])) + K.epsilon()), axis=(0, 1))

        regularized_loss = loss + pen1.sum() * self.TV
        regularized_loss += pen2.sum() * self.TV2

        if self.colors is 3:
            pen3 = K.sum(K.sqrt(K.square(diffr(p[1])) + K.square(diffc(p[1])) + K.epsilon()), axis=(0, 1))
            pen4 = K.sum(K.sqrt(K.square(diffrr(p[1])) + K.square(diffcc(p[1])) + 2*K.square(diffrc(p[1])) + K.epsilon()), axis=(0, 1))
            pen5 = K.sum(K.sqrt(K.square(diffr(p[2])) + K.square(diffc(p[2])) + K.epsilon()), axis=(0, 1))
            pen6 = K.sum(K.sqrt(K.square(diffrr(p[2])) + K.square(diffcc(p[2])) + 2*K.square(diffrc(p[2])) + K.epsilon()), axis=(0, 1))

            regularized_loss += pen3.sum() * self.TV
            regularized_loss += pen4.sum() * self.TV2
            regularized_loss += pen5.sum() * self.TV
            regularized_loss += pen6.sum() * self.TV2

        return K.in_train_phase(regularized_loss, loss)

    def get_config(self):
        return {"name": self.__class__.__name__,
                "TV": self.TV,
                "TV2": self.TV2}


class TVWeightRegularizer2d(Regularizer):
    def __init__(self, TV=0., TV2=0.):
        self.TV = TV
        self.TV2 = TV2
        self.uses_learning_phase = True

    def set_param(self, p):
        self.p = p

    def __call__(self, loss):
        if K.ndim(self.p) == 3:
            p = K.reshape(self.p, (1, self.p.shape[0], self.p.shape[1], self.p.shape[2]))

        pen1 = K.sum(K.sqrt(K.square(diffr(p[0])) + K.square(diffc(p[0])) + K.epsilon()), axis=(0, 1))
        pen2 = K.sum(K.sqrt(K.square(diffrr(p[0])) + K.square(diffcc(p[0])) + 2*K.square(diffrc(p[0])) + K.epsilon()), axis=(0, 1))

        regularized_loss = loss + pen1.sum() * self.TV
        regularized_loss += pen2.sum() * self.TV2

        if p.shape[0] is 3:
            pen3 = K.sum(K.sqrt(K.square(diffr(p[1])) + K.square(diffc(p[1])) + K.epsilon()), axis=(0, 1))
            pen4 = K.sum(K.sqrt(K.square(diffrr(p[1])) + K.square(diffcc(p[1])) + 2*K.square(diffrc(p[1])) + K.epsilon()), axis=(0, 1))
            pen5 = K.sum(K.sqrt(K.square(diffr(p[2])) + K.square(diffc(p[2])) + K.epsilon()), axis=(0, 1))
            pen6 = K.sum(K.sqrt(K.square(diffrr(p[2])) + K.square(diffcc(p[2])) + 2*K.square(diffrc(p[2])) + K.epsilon()), axis=(0, 1))

            regularized_loss += pen3.sum() * self.TV
            regularized_loss += pen4.sum() * self.TV2
            regularized_loss += pen5.sum() * self.TV
            regularized_loss += pen6.sum() * self.TV2

        return K.in_train_phase(regularized_loss, loss)

    def get_config(self):
        return {"name": self.__class__.__name__,
                "TV": self.TV,
                "TV2": self.TV2}


class TVRegularizer(Regularizer):
    def __init__(self, TV=0., TV2=0., axes=[0, 1]):
        self.TV = TV
        self.TV2 = TV2
        self.axes = list(axes)
        self.uses_learning_phase = True

    def set_param(self, p):
        self.p = p

    def __call__(self, loss):
        dimorder = self.axes + list(set(range(K.ndim(self.p))) - set(self.axes))
        p = K.permute_dimensions(self.p, dimorder)

        pen1 = K.sum(K.sqrt(K.square(diffr(p)) + K.square(diffc(p)) + K.epsilon()), axis=(0, 1))
        pen2 = K.sum(K.sqrt(K.square(diffrr(p)) + K.square(diffcc(p)) + 2*K.square(diffrc(p)) + K.epsilon()), axis=(0, 1))

        regularized_loss = loss + pen1.sum() * self.TV
        regularized_loss += pen2.sum() * self.TV2

        return K.in_train_phase(regularized_loss, loss)

    def get_config(self):
        return {"name": self.__class__.__name__,
                "TV": self.TV,
                "TV2": self.TV2}


def laplacian_l2(l=0.01):
    return LaplacianWeightRegularizer(l2=l)


def laplacian_l1(l=0.001):
    return LaplacianWeightRegularizer(l1=l)


def laplacian1d(X):
    return -2.0*X[1:-1] + X[0:-2] + X[2:]


def diffc(X):
    Xout = K.zeros(X.shape.eval())
    Xout = T.set_subtensor(Xout[:, 1:-1], X[:, 2:] - X[:, :-2])
    Xout = T.set_subtensor(Xout[:, 0], X[:, 1] - X[:, 0])
    Xout = T.set_subtensor(Xout[:, -1], X[:, -1] - X[:, -2])
    return Xout/2


def diffr(X):
    Xout = K.zeros(X.shape.eval())
    Xout = T.set_subtensor(Xout[1:-1, :], X[2:, :] - X[:-2, :])
    Xout = T.set_subtensor(Xout[0, :], X[1, :] - X[0, :])
    Xout = T.set_subtensor(Xout[-1, :], X[-1, :] - X[-2, :])
    return Xout/2


def diffcc(X):
    Xout = K.zeros(X.shape.eval())
    Xout = T.set_subtensor(Xout[:, 1:-1], X[:, 2:] + X[:, :-2])
    Xout = T.set_subtensor(Xout[:, 0], X[:, 1] + X[:, 0])
    Xout = T.set_subtensor(Xout[:, -1], X[:, -1] + X[:, -2])
    return Xout - 2*X


def diffrr(X):
    Xout = K.zeros(X.shape.eval())
    Xout = T.set_subtensor(Xout[1:-1, :], X[2:, :] + X[:-2, :])
    Xout = T.set_subtensor(Xout[0, :], X[1, :] + X[0, :])
    Xout = T.set_subtensor(Xout[-1, :], X[-1, :] + X[-2, :])
    return Xout - 2*X


def diffrc(X):
    Xout1 = K.zeros(X.shape.eval())
    Xout1 = T.set_subtensor(Xout1[1:-1, 1:-1], X[2:, 2:]+X[:-2, :-2])
    Xout1 = T.set_subtensor(Xout1[0, 0], X[1, 1]+X[0, 0])
    Xout1 = T.set_subtensor(Xout1[0, 1:-1], X[1, 2:]+X[0, :-2])
    Xout1 = T.set_subtensor(Xout1[0, -1], X[1, -1]+X[0, -2])
    Xout1 = T.set_subtensor(Xout1[1:-1, -1], X[2:, -1]+X[:-2, -2])
    Xout1 = T.set_subtensor(Xout1[-1, -1], X[-1, -1]+X[-2, -2])
    Xout1 = T.set_subtensor(Xout1[-1, 1:-1], X[-1, 2:]+X[-2, :-2])
    Xout1 = T.set_subtensor(Xout1[-1, 0], X[-1, 1]+X[-2, 0])
    Xout1 = T.set_subtensor(Xout1[1:-1, 0], X[2:, 1]+X[:-2, 0])

    Xout2 = K.zeros(X.shape.eval())
    Xout2 = T.set_subtensor(Xout2[1:-1, 1:-1], X[:-2, 2:]+X[2:, :-2])
    Xout2 = T.set_subtensor(Xout2[0, 0], X[0, 1]+X[1, 0])
    Xout2 = T.set_subtensor(Xout2[0, 1:-1], X[0, 2:]+X[1, :-2])
    Xout2 = T.set_subtensor(Xout2[0, -1], X[0, -1]+X[1, -2])
    Xout2 = T.set_subtensor(Xout2[1:-1, -1], X[:-2, -1]+X[2:, -2])
    Xout2 = T.set_subtensor(Xout2[-1, -1], X[-2, -1]+X[-1, -2])
    Xout2 = T.set_subtensor(Xout2[-1, 1:-1], X[-2, 2:]+X[-1, :-2])
    Xout2 = T.set_subtensor(Xout2[-1, 0], X[-2, 1]+X[-1, 0])
    Xout2 = T.set_subtensor(Xout2[1:-1, 0], X[:-2, 1]+X[2:, 0])

    return (Xout1 - Xout2)/4


from keras.utils.generic_utils import get_from_module
def get(identifier, kwargs=None):
    return get_from_module(identifier, globals(), 'regularizer', instantiate=True, kwargs=kwargs)

