#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
#*****************************************************************************
# > Author: foid
# > Mail: zw.ruan.sn@gmail.com 
# > Created Time: 11/23/2016
#******************************************************************************

import theano
import numpy as np

class CharEmbedding:
    def __init__(self, rng, config, n_char, emb = None):
        self.n_char = n_char
        self.dim = config.char_embedding_dim

        if emb is not None:
            emb_value = emb
        else:
            emb_value = rng.normal(loc = 0.0, scale = 0.01,
                    size = (n_char, self.dim))
        self.unigram_table = theano.shared(emb_value.astype(theano.config.floatX),
                name = 'char_embedding', borrow = True)

        self.params = [self.unigram_table]

        if config.use_bigram_feature:
            bigram_table = np.zeros((n_char * 2, self.dim),
                    dtype = theano.config.floatX)
            for i in xrange(n_char):
                for j in xrange(n_char):
                    bigram_table[i * n_char + j] = 0.5 * (emb_value[i] + emb_value[j])
            self.bigram_table = theano.shared(bigram_table, name = 'bigram_table', borrow = True)
            self.params.append(self.bigram_table)
