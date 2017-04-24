# -*- coding: utf-8 -*-

import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)

# from github.com/Lasagne/Lasagne/blob/master/lasagne/updates.py

def adagrad(loss, params, learning_rate = 1.0, epsilon = 1e-6):
    grads = theano.grad(loss, params)
    updates = OrderedDict()

    for param, grad in zip(params, grads):
        value = param.get_value(borrow = True)
        accu = theano.shared(np.zeros(value.shape, dtype = value.dtype),
                broadcastable = param.broadcastable)

        accu_new = accu + grad ** 2;
        updates[accu] = accu_new
        updates[param] = param - (learning_rate * grad /
                T.sqrt(accu_new + epsilon))

    return updates

def rmsprop(loss, params, learning_rate = 1.0, rho = 0.9, epsilon = 1e-6,
        clip = False):
    '''
    rho: Gradient moving average decay factor
    epsilon: Small value added for numerical stability
    '''
    grads = theano.grad(loss, params)
    if clip:
        grads = total_norm_constraint(grads)

    updates = OrderedDict()

    one = T.constant(1)

    for param, grad in zip(params, grads):
        value = param.get_value(borrow = True)
        accu_m = theano.shared(np.zeros(value.shape, dtype = value.dtype),
                broadcastable = param.broadcastable)
        accu_s = theano.shared(np.zeros(value.shape, dtype = value.dtype),
                broadcastable = param.broadcastable)

        accu_m_new = rho * accu_m + (one - rho) * grad
        accu_s_new = rho * accu_s + (one - rho) * grad ** 2

        updates[accu_s] = accu_s_new
        updates[accu_m] = accu_m_new
        updates[param] = param - (learning_rate * grad / \
                T.sqrt(accu_s_new - accu_m_new ** 2 + epsilon))

    return updates

def total_norm_constraint(param_list, max_norm = 5, epsilon = 1e-7):
    norm = T.sqrt(T.sum([T.sum(param ** 2) for  param in param_list]))
    dtype = np.dtype(theano.config.floatX).type

    target_norm = T.clip(norm, 0, dtype(max_norm))
    multiplier = target_norm / (dtype(epsilon) + norm)
    params_scaled = [param * multiplier for param in param_list]

    return params_scaled
