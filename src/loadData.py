# -*- coding: utf-8 -*-
"""
Created on Fri Apr 08 15:44:32 2016

@author: Lemon_Tree
"""

import numpy as np
import theano

OOV_KEY = '<OOV>'
BOS_KEY = '<BOS>'
EOS_KEY = '<EOS>'
#word_dict = {OOV_KEY:0, BOS_KEY:1, EOS_KEY:2}
tag2index = {'S':0, 'B':1, 'M':2, 'E':3}

def load_data(filename, char2index):
    data_ret = []
    label_ret = []
    with open(filename, 'r') as fin:
        for line in fin:
            line_items = line.decode('utf-8').strip('\r\n ').split(' ')
            if(len(line_items) < 5):
                continue

            sentence_char = []
            sentence_tag = []
            for char_tag in line_items:
                char, tag = char_tag.split('_')
                if char in char2index:
                    sentence_char.append(char2index[char])
                else:
                    sentence_char.append(char2index[OOV_KEY])
                sentence_tag.append(tag2index[tag])

            data_ret.append(sentence_char)
            label_ret.append(sentence_tag)
    return data_ret, label_ret

def words2window(x, left_n, right_n):
    x_with_padding = x
    # <BOS>: 1 
    if left_n != 0:
        padding_left = np.ones((x.shape[0], left_n), dtype = 'int32')
        x_with_padding = np.concatenate((padding_left, x_with_padding),
                axis = 1)

    # <EOS>: 2
    if right_n != 0:
        padding_right = np.ones((x.shape[0], right_n), dtype = 'int32') + 1
        x_with_padding = np.concatenate((x_with_padding, padding_right),
                axis = 1)
    # x_with_padding: (batch_size, left_n + x.shape[0] + right_n)

    # (n_step, batch, window_size)
    batch_training_data = np.zeros((x.shape[1], x.shape[0],
        left_n + 1 + right_n), dtype = 'int32')

    for index in range(left_n, left_n + x.shape[1]):
        # [index - left_n, index, index + right_n]
        window_word_index = x_with_padding[:, index - left_n:index + 1 + right_n]
        batch_training_data[index - left_n] = window_word_index

    return batch_training_data

def prepare_batch_data(seqs, labels, left_n, right_n):
    lengths = [len(s) for s in seqs]

    n_samples = len(seqs)
    maxlen = np.max(lengths)

    x = np.zeros((n_samples, maxlen), dtype = 'int32')
    x_mask = np.zeros((n_samples, maxlen), dtype = theano.config.floatX)
    y = -np.ones((n_samples, maxlen), dtype = 'int32')
    for idx, s in enumerate(seqs):
        x[idx, :lengths[idx]] = s
        x_mask[idx, :lengths[idx]] = 1.
        y[idx, :lengths[idx]] = labels[idx]

    x = words2window(x, left_n, right_n)

    return x, y.T, x_mask.T

def word2id(word_idx, data_x, data_y):
    new_data_x = []
    for sen in data_x:
        temp = []
        for word in sen:
            if word not in word_idx:
                temp.append(word_idx['<OOV>'])
            else:
                temp.append(word_idx[word])
        new_data_x.append(temp)
    tag_idx = ['S','B','E','M']
    new_data_y = []
    for tag in data_y:
        temp = []
        for t in tag:
            temp.append(tag_idx.index(t))
        new_data_y.append(temp)
    return new_data_x, new_data_y
