from __future__ import absolute_import
from keras import backend as K
import numpy as np
from keras.utils.generic_utils import get_from_module
from keras.optimizers import Optimizer
from six.moves import zip


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
        accum_switch = K.equal((self.iterations + 1.) % self.accum_iters, 0)
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