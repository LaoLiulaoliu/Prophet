#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuande Liu <miraclecome (at) gmail.com>

from __future__ import print_function, division

from word_vec import WordVec

TOTAL = 1626750
PROPORTION=0.7
BATCH = 50

def get_train_data():
    train_data_len = int(TOTAL * PROPORTION)
    train_iter_times = train_data_len // BATCH

    wordvec = WordVec()
    for bags in wordvec.generator_data(BATCH):
        yield bags
        train_iter_times -= 1
        if train_iter_times == 0: break


def get_validation_data():
    train_data_len = int(TOTAL * PROPORTION)
    train_iter_times = train_data_len // BATCH

    wordvec = WordVec()
    for bags in wordvec.generator_data(BATCH):
        train_iter_times -= 1
        if train_iter_times <= 0:
            yield bags


def get_data():
    pass

