from __future__ import absolute_import
from keras import backend as K
import numpy as np
from keras.utils.generic_utils import get_from_module
from six.moves import zip


def clip_norm(g, c, n):
    if c > 0:
        g = K.switch(n >= c, g * c / n, g)
    return g


def kl_divergence(p, p_hat):
    return p_hat - p + p * K.log(p / p_hat)


class Optimizer(object):
    '''Abstract optimizer base class.

    Note: this is the parent class of all optimizers, not an actual optimizer
    that can be used for training models.

    All Keras optimizers support the following keyword arguments:

        clipnorm: float >= 0. Gradients will be clipped
            when their L2 norm exceeds this value.
        clipvalue: float >= 0. Gradients will be clipped
            when their absolute value exceeds this value.
    '''
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.updates = []

    def get_state(self):
        return [K.get_value(u[0]) for u in self.updates]

    def set_state(self, value_list):
        assert len(self.updates) == len(value_list)
        for u, v in zip(self.updates, value_list):
            K.set_value(u[0], v)

    def get_updates(self, params, constraints, loss):
        raise NotImplementedError

    def get_gradients(self, loss, params):
        grads = K.gradients(loss, params)
        if hasattr(self, 'clipnorm') and self.clipnorm > 0:
            norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
            grads = [clip_norm(g, self.clipnorm, norm) for g in grads]
        if hasattr(self, 'clipvalue') and self.clipvalue > 0:
            grads = [K.clip(g, -self.clipvalue, self.clipvalue) for g in grads]
        return grads

    def get_config(self):
        return {"name": self.__class__.__name__}


class SGD(Optimizer):
    '''Stochastic gradient descent, with support for momentum,
    decay, and Nesterov momentum.

    # Arguments
        lr: float >= 0. Learning rate.
        momentum: float >= 0. Parameter updates momentum.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
    '''
    def __init__(self, lr=0.01, momentum=0., decay=0., nesterov=False,
                 *args, **kwargs):
        super(SGD, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.iterations = K.variable(0.)
        self.lr = K.variable(lr)
        self.momentum = K.variable(momentum)
        self.decay = K.variable(decay)

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        lr = self.lr * (1.0 / (1.0 + self.decay * self.iterations))
        self.updates = [(self.iterations, self.iterations + 1.)]

        for p, g, c in zip(params, grads, constraints):
            m = K.variable(np.zeros(K.get_value(p).shape))  # momentum
            v = self.momentum * m - lr * g  # velocity
            self.updates.append((m, v))

            if self.nesterov:
                new_p = p + self.momentum * v - lr * g
            else:
                new_p = p + v

            self.updates.append((p, c(new_p)))  # apply constraints
        return self.updates

    def get_config(self):
        return {"name": self.__class__.__name__,
                "lr": float(K.get_value(self.lr)),
                "momentum": float(K.get_value(self.momentum)),
                "decay": float(K.get_value(self.decay)),
                "nesterov": self.nesterov}


class RMSprop(Optimizer):
    '''RMSProp optimizer.

    It is recommended to leave the parameters of this optimizer
    at their default values.

    This optimizer is usually a good choice for recurrent
    neural networks.

    # Arguments
        lr: float >= 0. Learning rate.
        rho: float >= 0.
        epsilon: float >= 0. Fuzz factor.
    '''
    def __init__(self, lr=0.001, rho=0.9, epsilon=1e-6, momentum=0.5, decay=0., nesterov=False, adapt=.01, adapt_min=1., adapt_max=5., *args, **kwargs):
        super(RMSprop, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.lr = K.variable(lr)
        self.iterations = K.variable(0.)
        self.rho = K.variable(rho)
        self.decay = K.variable(decay)
        self.momentum = K.variable(momentum)
        self.adapt_min = adapt_min
        self.adapt_max = adapt_max
        self.adapt = K.variable(adapt)

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        lr = self.lr * (1.0 / (1.0 + self.decay * self.iterations))
        # accumulators = [K.variable(np.zeros(K.get_value(p).shape)) for p in params]
        self.weights = [K.variable(np.zeros(K.get_value(p).shape)) for p in params]
        self.updates = [(self.iterations, self.iterations + 1.)]

        for p, g, a in zip(params, grads, self.weights):
            m = K.variable(np.zeros(K.get_value(p).shape))  # momentum
            ada = K.variable(np.ones(K.get_value(p).shape))  # adaptation
            # update accumulator
            new_a = self.rho * a + (1 - self.rho) * K.square(g)
            self.updates.append((a, new_a))

            g /= K.sqrt(new_a + self.epsilon)

            new_adapt = K.clip(ada * (1.0 - self.adapt*K.sign(g)*K.sign(m)), self.adapt_min, self.adapt_max)
            self.updates.append((ada, new_adapt))

            v = self.momentum * m - new_adapt * lr * g # velocity
            self.updates.append((m, v))

            if self.nesterov:
                new_p = p + self.momentum * v - new_adapt * lr * g
            else:
                new_p = p + v

            # apply constraints
            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)
            self.updates.append((p, new_p))
        return self.updates

    def get_config(self):
        return {"name": self.__class__.__name__,
                "lr": float(K.get_value(self.lr)),
                "rho": float(K.get_value(self.rho)),
                "epsilon": self.epsilon,
                "momentum": float(K.get_value(self.momentum)),
                "decay": float(K.get_value(self.decay)),
                "adapt_min": float(K.get_value(self.adapt_min)),
                "adapt_max": float(K.get_value(self.adapt_max)),
                "adapt": float(K.get_value(self.adapt)),
                "nesterov": self.nesterov}


class Adagrad(Optimizer):
    '''Adagrad optimizer.

    It is recommended to leave the parameters of this optimizer
    at their default values.

    # Arguments
        lr: float >= 0. Learning rate.
        epsilon: float >= 0.
    '''
    def __init__(self, lr=0.01, epsilon=1e-6, *args, **kwargs):
        super(Adagrad, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.lr = K.variable(lr)

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        accumulators = [K.variable(np.zeros(K.get_value(p).shape)) for p in params]
        self.updates = []

        for p, g, a, c in zip(params, grads, accumulators, constraints):
            new_a = a + K.square(g)  # update accumulator
            self.updates.append((a, new_a))
            new_p = p - self.lr * g / K.sqrt(new_a + self.epsilon)
            self.updates.append((p, c(new_p)))  # apply constraints
        return self.updates

    def get_config(self):
        return {"name": self.__class__.__name__,
                "lr": float(K.get_value(self.lr)),
                "epsilon": self.epsilon}


class Adadelta(Optimizer):
    '''Adadelta optimizer.

    It is recommended to leave the parameters of this optimizer
    at their default values.

    # Arguments
        lr: float >= 0. Learning rate. It is recommended to leave it at the default value.
        rho: float >= 0.
        epsilon: float >= 0. Fuzz factor.

    # References
        - [Adadelta - an adaptive learning rate method](http://arxiv.org/abs/1212.5701)
    '''
    def __init__(self, lr=1.0, rho=0.95, epsilon=1e-6, *args, **kwargs):
        super(Adadelta, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.lr = K.variable(lr)

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        accumulators = [K.variable(np.zeros(K.get_value(p).shape)) for p in params]
        delta_accumulators = [K.variable(np.zeros(K.get_value(p).shape)) for p in params]
        self.updates = []

        for p, g, a, d_a, c in zip(params, grads, accumulators,
                                   delta_accumulators, constraints):
            # update accumulator
            new_a = self.rho * a + (1 - self.rho) * K.square(g)
            self.updates.append((a, new_a))

            # use the new accumulator and the *old* delta_accumulator
            update = g * K.sqrt(d_a + self.epsilon) / K.sqrt(new_a + self.epsilon)

            new_p = p - self.lr * update
            self.updates.append((p, c(new_p)))  # apply constraints

            # update delta_accumulator
            new_d_a = self.rho * d_a + (1 - self.rho) * K.square(update)
            self.updates.append((d_a, new_d_a))
        return self.updates

    def get_config(self):
        return {"name": self.__class__.__name__,
                "lr": float(K.get_value(self.lr)),
                "rho": self.rho,
                "epsilon": self.epsilon}


class Adam(Optimizer):
    '''Adam optimizer.

    Default parameters follow those provided in the original paper.

    # Arguments
        lr: float >= 0. Learning rate.
        beta_1/beta_2: floats, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor.

    # References
        - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
    '''
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.,
                 *args, **kwargs):
        super(Adam, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.iterations = K.variable(0)
        self.lr = K.variable(lr)
        self.beta_1 = K.variable(beta_1)
        self.beta_2 = K.variable(beta_2)
        self.decay = K.variable(decay)

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        self.updates = [(self.iterations, self.iterations+1.)]

        t = self.iterations + 1
        lr_t = self.lr * (1.0 / (1.0 + self.decay * self.iterations)) * K.sqrt(1 - K.pow(self.beta_2, t)) / (1 - K.pow(self.beta_1, t))

        for p, g, c in zip(params, grads, constraints):
            # zero init of moment
            m = K.variable(np.zeros(K.get_value(p).shape))
            # zero init of velocity
            v = K.variable(np.zeros(K.get_value(p).shape))

            m_t = (self.beta_1 * m) + (1 - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1 - self.beta_2) * K.square(g)
            p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append((m, m_t))
            self.updates.append((v, v_t))
            self.updates.append((p, c(p_t)))  # apply constraints
        return self.updates

    def get_config(self):
        return {"name": self.__class__.__name__,
                "lr": float(K.get_value(self.lr)),
                "beta_1": float(K.get_value(self.beta_1)),
                "beta_2": float(K.get_value(self.beta_2)),
                "decay": float(K.get_value(self.decay)),
                "epsilon": self.epsilon}


class Adamax(Optimizer):
    '''Adamax optimizer from Adam paper's Section 7. It is a variant
     of Adam based on the infinity norm.

    Default parameters follow those provided in the paper.

    # Arguments
        lr: float >= 0. Learning rate.
        beta_1/beta_2: floats, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor.

    # References
        - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
    '''
    def __init__(self, lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-8,
                 *args, **kwargs):
        super(Adamax, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.iterations = K.variable(0)
        self.lr = K.variable(lr)
        self.beta_1 = K.variable(beta_1)
        self.beta_2 = K.variable(beta_2)

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        self.updates = [(self.iterations, self.iterations+1.)]

        t = self.iterations + 1
        lr_t = self.lr / (1 - K.pow(self.beta_1, t))

        for p, g, c in zip(params, grads, constraints):
            # zero init of 1st moment
            m = K.variable(np.zeros(K.get_value(p).shape))
            # zero init of exponentially weighted infinity norm
            u = K.variable(np.zeros(K.get_value(p).shape))

            m_t = (self.beta_1 * m) + (1 - self.beta_1) * g
            u_t = K.maximum(self.beta_2 * u, K.abs(g))
            p_t = p - lr_t * m_t / (u_t + self.epsilon)

            self.updates.append((m, m_t))
            self.updates.append((u, u_t))
            self.updates.append((p, c(p_t)))  # apply constraints
        return self.updates

    def get_config(self):
        return {"name": self.__class__.__name__,
                "lr": float(K.get_value(self.lr)),
                "beta_1": float(K.get_value(self.beta_1)),
                "beta_2": float(K.get_value(self.beta_2)),
                "epsilon": self.epsilon}


class NadamAccum(Optimizer):
    '''
    Nesterov Adam optimizer: Much like Adam is essentially RMSprop with momentum,
    Nadam is Adam RMSprop with Nesterov momentum.

    Default parameters follow those provided in the paper.
    It is recommended to leave the parameters of this optimizer
    at their default values.

    # Arguments
        lr: float >= 0. Learning rate.
        beta_1/beta_2: floats, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor.

    # References
        - [Nadam report](http://cs229.stanford.edu/proj2015/054_report.pdf)
        - [On the importance of initialization and momentum in deep learning](http://www.cs.toronto.edu/~fritz/absps/momentum.pdf)
    '''
    def __init__(self, lr=0.002, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, schedule_decay=0.004, accum_iters=1, **kwargs):
        super(NadamAccum, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.iterations = K.variable(0.)
        self.m_schedule = K.variable(1.)
        self.lr = K.variable(lr)
        self.beta_1 = K.variable(beta_1)
        self.beta_2 = K.variable(beta_2)
        self.schedule_decay = schedule_decay
        self.accum_iters = K.variable(accum_iters)

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        t = (self.iterations + 1.)/self.accum_iters
        accum_switch = K.floor((self.accum_iters - K.mod(self.iterations + 1., self.accum_iters))/self.accum_iters)
        # Due to the recommendations in [2], i.e. warming momentum schedule
        momentum_cache_t = self.beta_1 * (1. - 0.5 * (K.pow(0.96, t * self.schedule_decay)))
        momentum_cache_t_1 = self.beta_1 * (1. - 0.5 * (K.pow(0.96, (t + 1) * self.schedule_decay)))
        m_schedule_new = self.m_schedule * momentum_cache_t
        m_schedule_next = self.m_schedule * momentum_cache_t * momentum_cache_t_1
        self.updates.append((self.m_schedule, accum_switch*m_schedule_new + (1-accum_switch)*self.m_schedule))

        shapes = [x.shape for x in K.batch_get_value(params)]
        ms = [K.zeros(shape) for shape in shapes]
        vs = [K.zeros(shape) for shape in shapes]
        gs = [K.zeros(shape) for shape in shapes]

        self.weights = [self.iterations] + ms + vs

        for p, gp, m, v, ga in zip(params, grads, ms, vs, gs):

            g = (ga + gp)/self.accum_iters
            # the following equations given in [1]
            g_prime = g / (1. - m_schedule_new)
            m_t = self.beta_1 * m + (1. - self.beta_1) * g
            m_t_prime = m_t / (1. - m_schedule_next)
            v_t = self.beta_2 * v + (1. - self.beta_2) * K.square(g)
            v_t_prime = v_t / (1. - K.pow(self.beta_2, t))
            m_t_bar = (1. - momentum_cache_t) * g_prime + momentum_cache_t_1 * m_t_prime

            self.updates.append(K.update(m, (1-accum_switch)*m + accum_switch*m_t))
            self.updates.append(K.update(v, (1-accum_switch)*v + accum_switch*v_t))
            self.updates.append(K.update(ga, (1-accum_switch)*(ga + gp)))

            p_t = p - self.lr * m_t_bar / (K.sqrt(v_t_prime) + self.epsilon)
            new_p = p_t

            # apply constraints
            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)
            self.updates.append(K.update(p, (1-accum_switch)*p + accum_switch*new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'epsilon': self.epsilon,
                  'schedule_decay': self.schedule_decay,
                  'accum_iters': self.accum_iters}
        base_config = super(NadamAccum, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# aliases
sgd = SGD
rmsprop = RMSprop
adagrad = Adagrad
adadelta = Adadelta
adam = Adam
adamax = Adamax


def get(identifier, kwargs=None):
    return get_from_module(identifier, globals(), 'optimizer',
                           instantiate=True, kwargs=kwargs)
