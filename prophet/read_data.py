#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuande Liu <miraclecome (at) gmail.com>

from __future__ import print_function, division

from word_vec import WordVec
import numpy as np

PROPORTION=0.7

def get_train_data():
    wordvec = WordVec()
    bag_words = np.mat(wordvec.prime_data())
    m, n = np.shape(bag_words)
    train_data_len = int(m * PROPORTION)
    validation_data_len = m - tran_data_len

    data_idx = np.arange(m)
    np.random.shuffle(data_idx)

    return bag_words[data_idx[:train_data_len], :], bag_words[data_idx[train_data_len:], :]

def get_validation_data():
    pass

def get_data():
    pass


train, validation = get_train_data()
print(train[:10])
