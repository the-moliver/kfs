from __future__ import absolute_import
from keras import backend as K
from keras.regularizers import Regularizer


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
        pen = .5*K.sum(K.sqr(K.dot(p.T, p)/p.shape[0]))
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
        pen = K.sum(K.sqr(K.dot(output.T, output)/output.shape[0]))
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
        pen = .5*K.sum(K.sqr(K.dot(output[:, :self.dividx].T, output[:, self.dividx:])/output.shape[0]))
        loss += self.gamma*pen
        return loss

    def get_config(self):
        return {"name": self.__class__.__name__,
                "gamma": self.gamma}


class LaplacianWeightRegularizer(Regularizer):
    def __init__(self, l1=0., l2=0.):
        self.l1 = l1
        self.l2 = l2

    def set_param(self, p):
        self.p = p

    def __call__(self, loss):
        loss += K.sum(abs(laplacian1d(self.p))) * self.l1
        loss += K.sum(laplacian1d(self.p) ** 2) * self.l2
        return loss

    def get_config(self):
        return {"name": self.__class__.__name__,
                "l1": self.l1,
                "l2": self.l2}


class TVWeightRegularizer(Regularizer):
    def __init__(self, TV=0., TV2=0., row=1, col=1):
        self.TV = TV
        self.TV2 = TV2
        self.row = row
        self.col = col

    def set_param(self, p):
        self.p = p

    def __call__(self, loss):
        p = K.reshape(self.p, (self.row, self.row, self.p.shape[1]))
        pen1 = K.sum(K.sqrt(K.square(diffr(p)) + K.sqr(diffc(p)) + K.epsilon()), axis=(0, 1))
        pen2 = K.sum(K.sqrt(K.square(diffrr(p)) + K.sqr(diffcc(p)) + 2*K.sqr(diffrc(p)) + K.epsilon()), axis=(0, 1))

        loss += pen1.sum() * self.TV
        loss += pen2.sum() * self.TV2
        return loss

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
    Xout = K.zeros(X.shape)
    Xout = K.set_subtensor(Xout[:,1:-1], X[:,2:] - X[:,:-2])
    Xout = K.set_subtensor(Xout[:,0], X[:,1] - X[:,0])
    Xout = K.set_subtensor(Xout[:,-1], X[:,-1] - X[:,-2])
    return Xout/2

def diffr(X):
    Xout = K.zeros(X.shape)
    Xout = K.set_subtensor(Xout[1:-1,:], X[2:,:] - X[:-2,:])
    Xout = K.set_subtensor(Xout[0,:], X[1,:] - X[0,:])
    Xout = K.set_subtensor(Xout[-1,:], X[-1,:] - X[-2,:])
    return Xout/2


def diffcc(X):
    Xout = K.zeros(X.shape)
    Xout = K.set_subtensor(Xout[:,1:-1], X[:,2:] + X[:,:-2])
    Xout = K.set_subtensor(Xout[:,0], X[:,1] + X[:,0])
    Xout = K.set_subtensor(Xout[:,-1], X[:,-1] + X[:,-2])
    return Xout - 2*X

def diffrr(X):
    Xout = K.zeros(X.shape)
    Xout = K.set_subtensor(Xout[1:-1,:], X[2:,:] + X[:-2,:])
    Xout = K.set_subtensor(Xout[0,:], X[1,:] + X[0,:])
    Xout = K.set_subtensor(Xout[-1,:], X[-1,:] + X[-2,:])
    return Xout - 2*X

def diffrc(X):
    Xout1 = K.zeros(X.shape)
    Xout1 = K.set_subtensor(Xout1[1:-1,1:-1], X[2:,2:]+X[:-2,:-2])
    Xout1 = K.set_subtensor(Xout1[0,0], X[1,1]+X[0,0])
    Xout1 = K.set_subtensor(Xout1[0,1:-1], X[1,2:]+X[0,:-2])
    Xout1 = K.set_subtensor(Xout1[0,-1], X[1,-1]+X[0,-2])
    Xout1 = K.set_subtensor(Xout1[1:-1,-1], X[2:,-1]+X[:-2,-2])
    Xout1 = K.set_subtensor(Xout1[-1,-1], X[-1,-1]+X[-2,-2])
    Xout1 = K.set_subtensor(Xout1[-1,1:-1], X[-1,2:]+X[-2,:-2])
    Xout1 = K.set_subtensor(Xout1[-1,0], X[-1,1]+X[-2,0])
    Xout1 = K.set_subtensor(Xout1[1:-1,0], X[2:,1]+X[:-2,0])


    Xout2 = K.zeros(X.shape)
    Xout2 = K.set_subtensor(Xout2[1:-1,1:-1], X[:-2,2:]+X[2:,:-2])
    Xout2 = K.set_subtensor(Xout2[0,0], X[0,1]+X[1,0])
    Xout2 = K.set_subtensor(Xout2[0,1:-1], X[0,2:]+X[1,:-2])
    Xout2 = K.set_subtensor(Xout2[0,-1], X[0,-1]+X[1,-2])
    Xout2 = K.set_subtensor(Xout2[1:-1,-1], X[:-2,-1]+X[2:,-2])
    Xout2 = K.set_subtensor(Xout2[-1,-1], X[-2,-1]+X[-1,-2])
    Xout2 = K.set_subtensor(Xout2[-1,1:-1], X[-2,2:]+X[-1,:-2])
    Xout2 = K.set_subtensor(Xout2[-1,0], X[-2,1]+X[-1,0])
    Xout2 = K.set_subtensor(Xout2[1:-1,0], X[:-2,1]+X[2:,0])

    return (Xout1 - Xout2)/4


from keras.utils.generic_utils import get_from_module
def get(identifier, kwargs=None):
    return get_from_module(identifier, globals(), 'regularizer', instantiate=True, kwargs=kwargs)

