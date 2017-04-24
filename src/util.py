#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
#*****************************************************************************
# > Author: foid
# > Mail: zw.ruan.sn@gmail.com 
# > Created Time: 06/24/2016
#******************************************************************************
import theano
import theano.tensor as T
import numpy as np

def len_argsort(seq):
    return sorted(range(len(seq)), key = lambda x: len(seq[x]))

def shared_dataset(data_x, data_y, borrow=True):
    shared_x = theano.shared(np.asarray(data_x, dtype = 'int32'),
            borrow = borrow)
    shared_y = theano.shared(np.asarray(data_y,
        dtype = theano.config.floatX), borrow = borrow)

    return shared_x, T.cast(shared_y, 'int32')

def shared_batch_data(x, mask, y, borrow=True):
    shared_x = theano.shared(np.asarray(x, dtype = theano.config.floatX),
            borrow = borrow)
    shared_mask = theano.shared(np.asarray(mask, dtype = theano.config.floatX),
            borrow = borrow)
    shared_y = theano.shared(np.asarray(y, dtype = theano.config.floatX),
            borrow = borrow)

    return T.cast(shared_x, 'int32'), shared_mask, T.cast(shared_y, 'int32')

def progress_bar_str(current, total):
    completed = int(float(current) / total * 50)
    remain = 50 - completed
    progress_bar = "[{:s}>{:s}] {:d}/{:d}".format('=' * completed,
            ' ' * remain, current, total)
    return progress_bar

def get_seg(data, label, index2char):
    ret = []
    for i in xrange(len(data)):
        line = []
        word = u''
        sen = data[i]
        res = label[i]
        # {'S':0, 'B':1, 'M':2, 'E':3}
        for j in xrange(len(sen)):
            if res[j] == 0:
                word = index2char[sen[j]]
                line.append(word)
                word = u''
            elif res[j] == 3:
                word += index2char[sen[j]]
                line.append(word)
                word = u''
            else:
                word += index2char[sen[j]]
        if len(word) != 0:
            line.append(word)
        ret.append(line)
    return ret

def evaluate(ans, pred):
    right = 0
    wrong = 0
    tot_right = 0
    for i in range(0,len(pred)):
        line1 = pred[i]
        line2 = ans[i]
        res1 = []
        res2 = []
        j = 0
        for word in line1:
            l = len(word)
            res1.append(l)
            for j in range(1,l):
                res1.append(-1)
        for word in line2:
            l = len(word)
            res2.append(l)
            for j in range(1,l):
                res2.append(-1)
        for j in range(0,len(res1)):
            if res1[j] == -1:
                continue
            if res1[j] == res2[j]:
                right += 1
            else:
                wrong += 1
        tot_right += len(line2)
        #print 'right=%d' % right
        #print 'wrong=%d' % wrong
    p = (1.0*right/(right+wrong))
    r = (1.0*right/tot_right)
    f = (2*p*r/(p+r))
    return (p, r, f)

def pred2list(pred, length):
    ret = []
    for index, l in enumerate(length):
        ret.append(pred[index][:l])

    return ret

def invert_char2index(char2index):
    index2char = {}
    for char, index in char2index.items():
        index2char[index] = char

    return index2char

def add_special_key(embedding, char2index, special_key, rng):
    '''
    special_key: a list contain special keys
    '''
    n_char = len(char2index) + len(special_key)
    new_embedding = rng.normal(loc = 0.0, scale = 0.01, size = (n_char,
        embedding.shape[1]))
    new_embedding[len(special_key):] = embedding

    new_char2index = {}
    for index, key in enumerate(special_key):
        new_char2index[key] = index
    for key in char2index.keys():
        new_char2index[key] = len(new_char2index)

    return new_embedding, new_char2index

def embedding_from_text(filename, dtype = theano.config.floatX):
    with open(filename, 'r') as fin:
        head = fin.readline().decode('utf-8').strip()
        n_char, dim = head.split()
        n_char, dim = int(n_char), int(dim)

        embedding = np.random.random((n_char, dim)).astype(dtype)
        char2index = {}
        for line in fin:
            ch, vec = line.decode('utf-8').strip().split(' ', 1)
            vec = np.fromstring(vec, dtype = dtype, sep = ' ')
            assert vec.shape[0] == dim

            embedding[len(char2index)] = vec
            char2index[ch] = len(char2index)

    return embedding, char2index

def save_model(params, path, name, suffix = ''):
    from theano.misc import pkl_utils
    with open(path + name + suffix, 'wb') as fout:
        for param in params:
            pkl_utils.dump(param.get_value(), fout)

def load_model(params, filename):
    from theano.misc import pkl_utils
    with open(filename, 'rb') as fin:
        for param in params:
            param.set_value(pkl_utils.load(fin))
