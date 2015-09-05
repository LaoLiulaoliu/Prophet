#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuande Liu <miraclecome (at) gmail.com>

from __future__ import print_function, division

import string
from collections import defaultdict


def load_words(filename='weibo_train_participle.txt'):
    words = defaultdict(dict)
    Y = []
    with open(filename) as fd:
        for idx, line in enumerate(fd):
            item = line.split('\t', 6)
            words[idx] = filter(lambda x: x != '', map(string.strip, item[-1].split(',')))
            Y.append(map(int, (item[3], item[4], item[5])))
    return words, Y

def unique_word(words):
    word_set = set()
    for i, word_list in words.iteritems():
        word_set.update(word_list)
    return list(word_set)

def bag_word(words, word_set_list, Y=None):
    word_set_len = len(word_set_list)
    bag_words = []

    for i, word_list in words.iteritems():
        one_bag = [0] * word_set_len
        for word in word_list:
            one_bag[word_set_list.index(word)] += 1
        if Y:
            one_bag.extend(Y[i])
        bag_words.append(one_bag)
    return bag_words

if __name__ == '__main__':
    words, Y = load_words('../data/weibo_train_participle.txt')
    word_set_list = unique_word(words)
    bag_words = bag_word(words, word_set_list, Y)
    print(bag_words[:10])

