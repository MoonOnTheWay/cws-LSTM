#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
#*****************************************************************************
# > Author: foid
# > Mail: zw.ruan.sn@gmail.com 
# > Created Time: 06/24/2016
#******************************************************************************

import logging
import sys
sys.path.append('../')
import time
import numpy as np

from src.config import Config
from src.loadData import load_data, prepare_batch_data
from src import util
from src.optimizer import adagrad
from src.embedding import CharEmbedding
from src.lstm import LSTMEncoder
from src.dropout import dropout_layer

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

#import pdb

logger = logging.getLogger(__name__)

# print the whole numpy array
np.set_printoptions(threshold=np.nan)

def config_logger(filename):
    log_fomatter = logging.Formatter(fmt='%(asctime)s [%(processName)s,'
            ' %(process)s] [%(levelname)s] %(message)s',
            datefmt = '%m-%d %H:%M')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler('../result/baseline/log/{:s}.log'.format(
        filename))
    file_handler.setFormatter(log_fomatter)
    file_handler.setLevel(logging.DEBUG)

    screen_handler = logging.StreamHandler()
    screen_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(screen_handler)

def model_name(config):
    model_str = ('Wemb{:d}_hidden{:d}_wl{:d}_wr{:d}_dropout{:f}_batch{:d}_'
            'lr{:f}_MaxEpoch{:d}_eta{:f}_lambda{:f}'.format(
                config.char_embedding_dim, config.hidden_size,
                config.window_size_l, config.window_size_r,
                config.dropout_rate, config.batch_size,
                config.learning_rate, config.n_epoch, config.eta,
                config.weight_lambda))

    return model_str

def build_model(rng, config, n_char, embedding = None):
    trng = RandomStreams(1992)

    # flag = 1 for training, 0 for testing
    flag = T.scalar(dtype = theano.config.floatX)

    # (n_steps, batch, window_size)
    x = T.tensor3('x', dtype = 'int32')
    # (n_steps, batch)
    mask = T.matrix('mask', dtype = theano.config.floatX)
    # (n_steps, batch)
    y = T.matrix('y', dtype = 'int32')

    embedding = CharEmbedding(rng, config, n_char, embedding)
    lstm_encoder = LSTMEncoder(rng, config)

    emb = embedding.unigram_table[x.flatten()].reshape((x.shape[0], x.shape[1],
        x.shape[2] * config.char_embedding_dim))

    if config.use_bigram_feature:
        index = x[:, :, :-1] * n_char + x[:, :, 1:]
        bigram_emb = embedding.bigram_table[index.flatten()].reshape((x.shape[0],
                x.shape[1], (x.shape[2] - 1) * config.char_embedding_dim))
        emb = T.concatenate([emb, bigram_emb], axis = 2)

    if config.use_dropout:
        emb = dropout_layer(emb, flag, config.dropout_rate, trng)

    # (n_steps, batch, hidden_size)
    hidden_state = lstm_encoder.forward(emb, mask)

    # (n_steps, batch, tag_size)
    score = T.dot(hidden_state, lstm_encoder.W) + lstm_encoder.b[None, None, :]

    y_pred = lstm_encoder.viterbi_search(mask, score, y, flag)
    y_pred_score = lstm_encoder.get_y_score(score, y_pred, mask)
    y_score = lstm_encoder.get_y_score(score, y, mask)

    #margin_loss = theano.gradient.disconnected_grad(T.sum(T.neq(y_pred,
    #    y) * mask, axis = 0) * self.config.eta)
    #margin_loss = theano.printing.Print('margin_loss:')(margin_loss)

    params = []
    for module in [embedding, lstm_encoder]:
        params.extend(module.params)

    weight_decay = T.sum([T.sum(param ** 2) for param in params])

    #loss = y_pred_score + margin_loss - y_score
    loss = y_pred_score  - y_score
    #cost = T.switch(T.gt(loss, 0),
    #        loss, T.zeros_like(loss, dtype = theano.config.floatX))

    cost = loss.mean() + config.weight_lambda * weight_decay

    learning_rate = T.scalar('learnig rate')
    #grads = theano.grad(cost, self.params)
    #updates = OrderedDict()
    #for grad, param in zip(grads, self.params):
    #    updates[param] = param - learning_rate * grad
    updates = adagrad(cost, params, learning_rate)
    #updates = rmsprop(cost, self.params, learning_rate, clip = True)

    train = theano.function([x, y, mask, learning_rate, flag], cost, updates = updates)
    test = theano.function([x, y, mask, flag], y_pred)

    return train, test, params

def train_validate_test(config_file, verbose = True):
    # debug data
    #train_file = '../data/pku_noNumnoEngnoIdiom/with_label/pku_train.small'
    #dev_file = '../data/pku_noNumnoEngnoIdiom/with_label/pku_dev.small'
    #test_file = '../data/pku_noNumnoEngnoIdiom/with_label/pku_test.small'
    embedding_file = '../data/pku_noNumnoEngnoIdiom/raw/pku_training.utf8.min20.dim100.txt'
    # train
    train_file = '../data/pku_noNumnoEngnoIdiom/with_label/pku_train.txt'
    dev_file = '../data/pku_noNumnoEngnoIdiom/with_label/pku_dev.txt'
    test_file = '../data/pku_noNumnoEngnoIdiom/with_label/pku_test.txt'
    config = Config(config_file)

    rng = np.random.RandomState(config.random_seed)

    embedding, char2index = util.embedding_from_text(embedding_file)
    special_key = ['<OOV>', '<BOS>', '<EOS>']
    embedding, char2index = util.add_special_key(embedding, char2index,
            special_key, rng)

    model_train, model_test, model_params = build_model(rng, config, len(char2index), embedding)

    train_x, train_y = load_data(train_file, char2index)
    test_x, test_y = load_data(test_file, char2index)

    if dev_file is not None:
        dev_x, dev_y = load_data(dev_file, char2index)
    else:
        # Spliting trianing data into train set and validation set
        train_end = np.int32(len(train_x) * 0.9)
        dev_x, dev_y = train_x[train_end:], train_y[train_end:]
        train_x, train_y = train_x[:train_end], train_y[:train_end]

    sorted_index = util.len_argsort(train_x)
    train_x = [train_x[i] for i in sorted_index]
    train_y = [train_y[i] for i in sorted_index]

    sorted_index = util.len_argsort(dev_x)
    dev_x = [dev_x[i] for i in sorted_index]
    dev_y = [dev_y[i] for i in sorted_index]

    sorted_index = util.len_argsort(test_x)
    test_x = [test_x[i] for i in sorted_index]
    test_y = [test_y[i] for i in sorted_index]

    index2char = util.invert_char2index(char2index)

    #pdb.set_trace()

    model_identifier_str = model_name(config)
    config_logger(model_identifier_str)

    batch_number = int(np.ceil(len(train_x) / config.batch_size))

    time_train_begin = time.time()

    best_epoch = -1
    best_f_score = 0.0
    for epoch_index in range(config.n_epoch):
        print('==================== Epoch {:d} =================='
              '=='.format(epoch_index + 1))

        ##############
        #  Training  #
        ##############
        time_epoch_begin = time.time()
        # generate random batch index
        batch_index_set = rng.permutation(batch_number)

        loss_history = []

        print('====> Training')
        for batch_count, batch_index in enumerate(batch_index_set):
            if verbose:
                print(util.progress_bar_str(batch_count, batch_number) + '\r'),
            train_batch_start = batch_index * config.batch_size
            train_batch_end = (batch_index + 1) * config.batch_size
            train_batch_end = min(train_batch_end, len(train_x))

            train_batch_x = train_x[train_batch_start:train_batch_end]
            train_batch_y = train_y[train_batch_start:train_batch_end]

            x, y, mask = prepare_batch_data(train_batch_x, train_batch_y,
                    config.window_size_l, config.window_size_r)
            #print(y.T)
            #pdb.set_trace()
            loss = model_train(x, y, mask, config.learning_rate, 1.0)
            #x,mask, y = shared_batch_data(x, mask, y)
            #loss = encoder.train(x.get_value(borrow = True),
            #                     y.get_value(borrow = True),
            #                     mask.get_value(borrow = True),
            #                     config.learning_rate)
            loss_history.append(loss)
        print('')
        util.save_model(model_params, '../result/baseline/model/', model_identifier_str,
                '_epoch{:d}.model'.format(epoch_index + 1))
        time_epoch_train_end = time.time()

        ##############
        # Validating #
        ##############
        pred_result = []
        dev_batch_number = int(np.ceil(len(dev_x) * 1.0 / config.batch_size))
        print('====> Dev')
        for dev_batch_index in range(dev_batch_number):
            if verbose:
                print(util.progress_bar_str(dev_batch_index, dev_batch_number) +
                        '\r'),
            dev_batch_start = dev_batch_index * config.batch_size
            dev_batch_end = (dev_batch_index + 1) * config.batch_size
            dev_batch_end = min(len(dev_x), dev_batch_end)

            dev_batch_x = dev_x[dev_batch_start:dev_batch_end]
            dev_batch_y = dev_y[dev_batch_start:dev_batch_end]
            batch_len = [len(item) for item in dev_batch_x]

            x, y, mask = prepare_batch_data(dev_batch_x, dev_batch_y,
                    config.window_size_l, config.window_size_r)
            y_pred = model_test(x, y, mask, 0.)
            #x, mask, y = shared_batch_data(x, mask, y)
            #correct, total = encoder.test(x.get_value(borrow=True),
            #y.get_value(borrow=True), mask.get_value(borrow=True))
            pred_result.extend(util.pred2list(y_pred.T, batch_len))
        print('')

        pred_seg = util.get_seg(dev_x, pred_result, index2char)
        ans_seg = util.get_seg(dev_x, dev_y, index2char)
        dev_result = util.evaluate(ans_seg, pred_seg)

        time_epoch_end = time.time()
        logger.info('train: {:f}, validate: {:f}'.format(time_epoch_train_end - \
                time_epoch_begin, time_epoch_end - time_epoch_train_end))
        logger.info('training loss: {:f}'.format(np.asarray(loss_history).mean()))
        logger.info('socre on dev: P: {:f}  R: {:f}  F: {:f}'.format(*dev_result))
        logger.info('')

        if dev_result[-1] > best_f_score:
            best_f_score = dev_result[-1]
            best_epoch = epoch_index


        ##############
        #    test    #
        ##############
        if (epoch_index + 1) % 10 == 0  or (epoch_index + 1) == config.n_epoch:
            print('====> Test')
            test_result = []
            test_batch_number = int(np.ceil(len(test_x) * 1.0 / config.batch_size))
            time_test_begin = time.time()
            for test_batch_index in range(test_batch_number):
                if verbose:
                    print(util.progress_bar_str(test_batch_index, test_batch_number) +
                            '\r'),
                test_batch_start = test_batch_index * config.batch_size
                test_batch_end = (test_batch_index + 1) * config.batch_size
                test_batch_end = min(test_batch_end, len(test_x))

                test_batch_x = test_x[test_batch_start:test_batch_end]
                test_batch_y = test_y[test_batch_start:test_batch_end]
                batch_len = [len(item) for item in test_batch_x]

                x, y, mask = prepare_batch_data(test_batch_x, test_batch_y,
                        config.window_size_l, config.window_size_r)
                pred =  model_test(x, y, mask, 0.)
                #x, mask, y = shared_batch_data(x, mask, y)
                #correct, total = encoder.test(x.get_value(borrow = True),
                #                              y.get_value(borrow = True),
                #                              mask.get_value(borrow = True))
                test_result.extend(util.pred2list(pred.T, batch_len))
            print('')

            pred_seg = util.get_seg(test_x, test_result, index2char)
            ans_seg = util.get_seg(test_x, test_y, index2char)
            test_score = util.evaluate(ans_seg, pred_seg)

            time_test_end = time.time()
            logger.info('time {:f}'.format(time_test_end - time_test_begin))
            logger.info('socre of test: P: {:f}  R: {:f}  F: {:f}'.format(*test_score))
            logger.info('')

    time_train_end = time.time()
    logger.info('================== Training End ==================')
    logger.info('time: {:f}'.format(time_train_end - time_train_begin))
    logger.info('')
    logger.info('================== Testing ====================')
    ##############
    # Testing #
    ##############
    test_result = []
    test_batch_number = int(np.ceil(len(test_x) * 1.0 / config.batch_size))
    logger.info('Using model from epoch {:d}'.format(best_epoch + 1))
    time_test_begin = time.time()
    util.load_model(model_params, '../result/baseline/model/' +  model_identifier_str +
            '_epoch{:d}.model'.format(best_epoch + 1))
    for test_batch_index in range(test_batch_number):
        if verbose:
            print(util.progress_bar_str(test_batch_index, test_batch_number) +
                    '\r'),
        test_batch_start = test_batch_index * config.batch_size
        test_batch_end = (test_batch_index + 1) * config.batch_size
        test_batch_end = min(test_batch_end, len(test_x))

        test_batch_x = test_x[test_batch_start:test_batch_end]
        test_batch_y = test_y[test_batch_start:test_batch_end]
        batch_len = [len(item) for item in test_batch_x]

        x, y, mask = prepare_batch_data(test_batch_x, test_batch_y,
                config.window_size_l, config.window_size_r)
        pred =  model_test(x, y, mask, 0.)
        #x, mask, y = shared_batch_data(x, mask, y)
        #correct, total = encoder.test(x.get_value(borrow = True),
        #                              y.get_value(borrow = True),
        #                              mask.get_value(borrow = True))
        test_result.extend(util.pred2list(pred.T, batch_len))
    print('')

    pred_seg = util.get_seg(test_x, test_result, index2char)
    ans_seg = util.get_seg(test_x, test_y, index2char)
    test_score = util.evaluate(ans_seg, pred_seg)

    time_test_end = time.time()
    logger.info('time {:f}'.format(time_test_end - time_test_begin))
    logger.info('socre of test: P: {:f}  R: {:f}  F: {:f}'.format(*test_score))
    logger.info('')

if __name__ == '__main__':
    config_file = '../conf/baseline/baseline.conf'
    train_validate_test(config_file, verbose = True)
