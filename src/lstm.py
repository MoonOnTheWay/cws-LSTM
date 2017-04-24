#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
#*****************************************************************************
# > Author: foid
# > Mail: zw.ruan.sn@gmail.com 
# > Created Time: 06/21/2016
#******************************************************************************

import numpy as np
import theano
import theano.tensor as T


def numpy_floatX(data):
    return np.asarray(data, dtype = theano.config.floatX)

class LSTMEncoder(object):
    '''
    according to http://www.deeplearning.net/tutorial/lstm.html#lstm
    '''
    def __init__(self, rng, config):
        '''
        rng: random state generator
        WordEmbedding: pretrained word embedding
        '''
        self.config = config

        W_value = rng.uniform(low = -np.sqrt(6. / (config.hidden_size + config.tag_size)),
                high = np.sqrt(6. / (config.hidden_size + config.tag_size)),
                size = (config.hidden_size, config.tag_size))
        self.W = theano.shared(W_value.astype(theano.config.floatX), name = 'W', borrow = True)

        b_value = np.zeros((config.tag_size,))
        self.b = theano.shared(b_value.astype(theano.config.floatX), name = 'b', borrow = True)

        if config.use_bigram_feature:
            input_size = config.char_embedding_dim * ((config.window_size_l +
                    1 + config.window_size_r) * 2 - 1)
        else:
            input_size = config.char_embedding_dim * (config.window_size_l +
                    1 + config.window_size_r)

        #####  input gate ######
        lstm_input_wx_value = rng.uniform(low = -np.sqrt(6. / (input_size + config.hidden_size)),
                high = np.sqrt(6. / (input_size + config.hidden_size)),
                size = (input_size, config.hidden_size)).astype(theano.config.floatX)
        self.lstm_input_wx = theano.shared(lstm_input_wx_value,
                name = 'lstm_input_wx', borrow = True)

        lstm_input_wh_value = rng.uniform(low = -np.sqrt(6. / (config.hidden_size * 2)),
            high = np.sqrt(6. / (config.hidden_size * 2)),
            size = (config.hidden_size, config.hidden_size)).astype(theano.config.floatX)
        self.lstm_input_wh = theano.shared(lstm_input_wh_value,
                name = 'lstm_input_wh', borrow = True)

        lstm_input_wc_value = rng.uniform(low = -np.sqrt(6. / (config.hidden_size * 2)),
                high = np.sqrt(6. / (config.hidden_size * 2)),
                size = (config.hidden_size, config.hidden_size))[0].astype(theano.config.floatX)
        self.lstm_input_wc = theano.shared(lstm_input_wc_value,
                name = 'lstm_input_wc', borrow = True)

        input_b_value = np.zeros((config.hidden_size,), dtype=theano.config.floatX)
        self.lstm_input_b = theano.shared(input_b_value, name = 'lstm_input_b', borrow = True)

        ##### forget gate #####
        lstm_forget_wx_value = rng.uniform(low = -np.sqrt(6. / (input_size + config.hidden_size)),
                high = np.sqrt(6. / (input_size + config.hidden_size)),
                size = (input_size, config.hidden_size)).astype(theano.config.floatX)
        self.lstm_forget_wx = theano.shared(lstm_forget_wx_value,
                name = 'lstm_forget_wx', borrow = True)

        lstm_forget_wh_value = rng.uniform(low = -np.sqrt(6. / (config.hidden_size * 2)),
            high = np.sqrt(6. / (config.hidden_size * 2)),
            size = (config.hidden_size, config.hidden_size)).astype(theano.config.floatX)
        self.lstm_forget_wh = theano.shared(lstm_forget_wh_value,
                name = 'lstm_forget_wh', borrow = True)

        lstm_forget_wc_value = rng.uniform(low = -np.sqrt(6. / (config.hidden_size * 2)),
                high = np.sqrt(6. / (config.hidden_size * 2)),
                size = (config.hidden_size, config.hidden_size))[0].astype(theano.config.floatX)
        self.lstm_forget_wc = theano.shared(lstm_forget_wc_value,
                name = 'lstm_forget_wc', borrow = True)

        forget_b_value = np.zeros((config.hidden_size,), dtype=theano.config.floatX)
        self.lstm_forget_b = theano.shared(forget_b_value, name = 'lstm_forget_b', borrow = True)

        ##### output gate #####
        lstm_output_wx_value = rng.uniform(low = -np.sqrt(6. / (input_size + config.hidden_size)),
                high = np.sqrt(6. / (input_size + config.hidden_size)),
                size = (input_size, config.hidden_size)).astype(theano.config.floatX)
        self.lstm_output_wx = theano.shared(lstm_output_wx_value,
                name = 'lstm_output_wx', borrow = True)

        lstm_output_wh_value = rng.uniform(low = -np.sqrt(6. / (config.hidden_size * 2)),
            high = np.sqrt(6. / (config.hidden_size * 2)),
            size = (config.hidden_size, config.hidden_size)).astype(theano.config.floatX)
        self.lstm_output_wh = theano.shared(lstm_output_wh_value,
                name = 'lstm_output_wh', borrow = True)

        lstm_output_wc_value = rng.uniform(low = -np.sqrt(6. / (config.hidden_size * 2)),
                high = np.sqrt(6. / (config.hidden_size * 2)),
                size = (config.hidden_size, config.hidden_size))[0].astype(theano.config.floatX)
        self.lstm_output_wc = theano.shared(lstm_output_wc_value,
                name = 'lstm_output_wc', borrow = True)

        output_b_value = np.zeros((config.hidden_size,), dtype=theano.config.floatX)
        self.lstm_output_b = theano.shared(output_b_value, name = 'lstm_forget_b', borrow = True)

        ##### cell tidel #####
        lstm_cell_wx_value = rng.uniform(low = -np.sqrt(6. / (input_size + config.hidden_size)),
                high = np.sqrt(6. / (input_size + config.hidden_size)),
                size = (input_size, config.hidden_size)).astype(theano.config.floatX)
        self.lstm_cell_wx = theano.shared(lstm_cell_wx_value,
                name = 'lstm_cell_wx', borrow = True)

        lstm_cell_wh_value = rng.uniform(low = -np.sqrt(6. / (config.hidden_size * 2)),
                high = np.sqrt(6. / (config.hidden_size * 2)),
                size = (config.hidden_size, config.hidden_size)).astype(theano.config.floatX)
        self.lstm_cell_wh = theano.shared(lstm_cell_wh_value,
                name = 'lstm_cell_wh', borrow = True)

        cell_b_value = np.zeros((config.hidden_size,), dtype = theano.config.floatX)
        self.lstm_cell_b = theano.shared(cell_b_value, name = 'lstm_cell_b', borrow = True)

        # Viterbi decoding parameters
        A_value = rng.normal(loc = 0.0, scale = 0.01,
                size = (config.tag_size, config.tag_size))
        self.A = theano.shared(A_value.astype(theano.config.floatX), name = 'A', borrow = True)

        self.params = [self.W, self.b, self.A,
                self.lstm_input_wx, self.lstm_input_wh, self.lstm_input_wc, self.lstm_input_b,
                self.lstm_forget_wx, self.lstm_forget_wh, self.lstm_forget_wc, self.lstm_forget_b,
                self.lstm_output_wx, self.lstm_output_wh, self.lstm_output_wc, self.lstm_output_b,
                self.lstm_cell_wx, self.lstm_cell_wh, self.lstm_cell_b]

    def step(self, mask, input_term, forget_term, output_term, cell_term, h_pre, c_pre):
        input_term += T.dot(h_pre, self.lstm_input_wh) + T.dot(c_pre, T.diag(self.lstm_input_wc))
        forget_term += T.dot(h_pre, self.lstm_forget_wh) + T.dot(c_pre, T.diag(self.lstm_forget_wc))

        input_term += self.lstm_input_b
        forget_term += self.lstm_forget_b

        input_gate = T.nnet.sigmoid(input_term)
        forget_gate = T.nnet.sigmoid(forget_term)

        cell_term += T.dot(h_pre, self.lstm_cell_wh)
        cell_term += self.lstm_cell_b

        c = forget_gate * c_pre + input_gate * T.tanh(cell_term)


        output_term += T.dot(h_pre, self.lstm_output_wh) + T.dot(c, T.diag(self.lstm_output_wc))
        output_term += self.lstm_output_b

        output_gate = T.nnet.sigmoid(output_term)

        h = output_gate * T.tanh(c)

        return h, c

    def forward(self, sentences, mask):
        '''
        sentences: (n_steps, batch_size, input_size)
        '''
        n_steps = sentences.shape[0]
        batch_size = sentences.shape[1]

        input_term = T.dot(sentences, self.lstm_input_wx)
        forget_term = T.dot(sentences, self.lstm_forget_wx)
        output_term = T.dot(sentences, self.lstm_output_wx)
        cell_term = T.dot(sentences, self.lstm_cell_wx)

        hidden_size= self.config.hidden_size
        # rval[0]: h --> (n_steps, batch_size, hidden_size)
        # rval[1]: c --> (n_steps, batch_size, hidden_size)
        rval, updates = theano.scan(self.step,
                sequences = [mask, input_term, forget_term, output_term, cell_term],
                outputs_info = [T.alloc(numpy_floatX(0.), batch_size, hidden_size),
                    T.alloc(numpy_floatX(0.), batch_size, hidden_size)],
                name = 'lstm_layers',
                n_steps = n_steps)

        return rval[0]

    def viterbi_step(self, score, mask, current_y, theta_prev, flag):
        """
        score: (batch, tag_size)
        theta_prev: (batch, tag_size)
        mask: (batch, )
        """
        # (batch, tag_size, tag_size) (batch, tag_size, tag_size)
        theta_candidate = theta_prev[:, :, None] + self.A[None, :, :]

        # constraint the tag transition, we mask the impossible transitions
        index1 = [0, 0, 1, 1, 2, 2, 3, 3]
        index2 = [2, 3, 0, 1, 0, 1, 2, 3]
        theta_candidate = T.set_subtensor(theta_candidate[:, index1, index2], -np.infty)

        #theta_candidate = theano.printing.Print('theta_candidate')(theta_candidate)
        # (batch, tag_size), (batch, tag_size)
        theta, index = T.max_and_argmax(theta_candidate, axis = 1)

        theta = theta + score

        plus_term = T.zeros_like(theta, dtype = theano.config.floatX)
        plus_term = plus_term + self.config.eta * flag
        plus_term = T.set_subtensor(plus_term[T.arange(plus_term.shape[0]), current_y], 0)
        theta = theta + plus_term

        theta = theta * mask[:, None] + (1 - mask)[:, None] * theta_prev

        condition = mask[:, None].repeat(index.shape[1], axis = 1)
        index = T.switch(condition, index,
                theta_prev.argmax(axis = 1)[:, None].repeat(index.shape[1],
                    axis = 1))

        return theta, index

    # length = 1 ???
    # so far, we filter out the sentences whose length are less than 5
    def viterbi_search(self, mask, score, y, flag):
        '''
        mask: (n_steps, batch_size)
        score: (n_steps, batch_size, tag_size)
        '''
        # (batch_size, tag_size)
        theta_initial = self.A[0][None, :] + score[0]

        # 复旦的代码中并没有对第一个符号进行限制
        #theta_initial = T.set_subtensor(theta_initial[:, 2:],
        #        T.zeros((score.shape[1], 2), dtype = theano.config.floatX))

        plus_term = T.zeros_like(theta_initial, dtype = theano.config.floatX)
        plus_term = plus_term + self.config.eta * flag
        plus_term = T.set_subtensor(plus_term[T.arange(plus_term.shape[0]), y[0]], 0)
        theta_initial = theta_initial + plus_term
        #theta_initial = theta_initial + self.config.eta
        #theta_initial[T.arange(theta_initial.shape[0]), y[0]] -= self.config.eta

        result, _ = theano.scan(self.viterbi_step,
                sequences = [score[1:], mask[1:], y[1:]],
                outputs_info = [theta_initial, None],
                non_sequences = [flag],
                name = 'viterbi search')

        theta, invert_pointer = result

        max_score, last_state = T.max_and_argmax(theta[-1], axis = 1)

        # (nsteps - 1, batch)
        rvert_tags, _ = theano.scan(lambda a, index: a[T.arange(a.shape[0]),
            index], sequences = [invert_pointer[::-1]],
                outputs_info = [last_state],
                name = 'rpointers to tag')

        tags_pred = T.zeros((rvert_tags.shape[0] + 1, rvert_tags.shape[1]),
                dtype = 'int32')
        tags_pred = T.set_subtensor(tags_pred[:-1], rvert_tags[::-1])
        tags_pred = T.set_subtensor(tags_pred[-1], last_state)

        #return max_score, tags_pred
        return tags_pred

    def y_score_step(self, score, y, mask, pre_y, score_pre):
        '''
        score: (batch, tag_size)
        y: (batch, )
        mask: (batch, )
        pre_y: (batch, )
        score_pre: (batch, )
        '''
        return score_pre + (self.A[pre_y, y] + score[T.arange(y.shape[0]), y]) * mask

    def get_y_score(self, score, y, mask):
        '''
        score: (n_steps, batch, tag_size)
        y: (n_steps, batch)
        mask: (n_steps, batch)
        '''
        # (batch, ) <-- (batch, ) + (batch, )
        score_pre = score[0, T.arange(y.shape[1]), y[0]] + self.A[0][y[0]]
        result, _ = theano.scan(self.y_score_step,
                sequences = [score[1:], y[1:], mask[1:], y[:-1]],
                    outputs_info = [score_pre],
                    name = 'get_y_score')

        return result[-1]


