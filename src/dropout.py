#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
#*****************************************************************************
# > Author: foid
# > Mail: zw.ruan.sn@gmail.com 
# > Created Time: 11/23/2016
#******************************************************************************
import theano.tensor as T

def dropout_layer(state_before, use_dropout, dropout_rate, rng):
    value = T.switch(use_dropout,
            state_before * rng.binomial(state_before.shape,
                p = 1 - dropout_rate, n = 1, dtype = state_before.dtype),
            state_before)

    return value
