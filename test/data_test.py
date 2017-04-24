#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
#*****************************************************************************
# > Author: foid
# > Mail: zw.ruan.sn@gmail.com 
# > Created Time: 07/27/2016
#******************************************************************************

import pdb
import sys
sys.path.append('../')

from src.loadData import loadTrainingData, words2window, prepare_batch_data

train_data = '../data/pku_noNumnoEngnoIdiom/with_label/pku_train.small'

x_train, y_train, word_idx, _ = loadTrainingData(train_data)
#x_train = [[20, 16, 18, 19, 60],
#        [100, 12, 13, 22, 25],
#        [0, 17, 111, 232, 113]]
#y_train = [[0, 1, 2, 3, 0],
#        [1, 2, 2, 3, 0],
#        [0, 0, 1, 2, 3]]

batch_data_x, batch_data_y = x_train[:3], y_train[:3]
batch_x, batch_y, batch_mask = prepare_batch_data(batch_data_x, batch_data_y,
        0, 2)
pdb.set_trace()
