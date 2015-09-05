#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuande Liu <miraclecome (at) gmail.com>

from __future__ import print_function, division

import string
from collections import defaultdict


class WordVec(object):
    def __init__(self, filename=None):
        if filename is None:
            filename = '../data/weibo_train_participle.txt'
        self.filename = filename

    def load_words(self):
        words = defaultdict(dict)
        Y = []
        with open(self.filename) as fd:
            for idx, line in enumerate(fd):
                item = line.split('\t', 6)
                words[idx] = filter(lambda x: x != '', map(string.strip, item[-1].split(',')))
                Y.append(map(int, (item[3], item[4], item[5])))
        return words, Y

    def unique_word(self, words):
        word_dict = defaultdict(dict)
        for i, word_list in words.iteritems():
            for word in word_list:
                if word not in word_dict:
                    word_dict[word] = len(word_dict)
        return word_dict

    def bag_word(self, words, word_dict, Y=None):
        word_set_len = len(word_dict)
        bag_words = []

        for i, word_list in words.iteritems():
            one_bag = [0] * word_set_len
            for word in word_list:
                one_bag[word_dict[word]] += 1
            if Y:
                one_bag.extend(Y[i])
            bag_words.append(one_bag)
        return bag_words

    def prime_data(self):
        words, Y = self.load_words()
        word_dict = self.unique_word(words)
        bag_words = self.bag_word(words, word_dict, Y)
        return bag_words



    def generator_bag_word(self, words, word_dict, Y=None):
        word_set_len = len(word_dict)

        for i, word_list in words.iteritems():
            one_bag = [0] * word_set_len
            for word in word_list:
                one_bag[word_dict[word]] += 1
            if Y:
                one_bag.extend(Y[i])
            yield one_bag

    def generator_data(self, batch_num=10):
        words, Y = self.load_words()
        word_dict = self.unique_word(words)
        one_bag = self.generator_bag_word(words, word_dict, Y)

        finish_flag = 0
        while finish_flag == 0:
            bags = []
            for i in range(batch_num):
                try:
                    one = one_bag.next()
                    bags.append(one)
                except:
                    finish_flag = 1
                    if bags != []: yield bags
                    break
            else:
                yield bags


if __name__ == '__main__':
    obj = WordVec('../data/weibo_train_participle.txt')
    bags = obj.generator_data()
    bags.next()

