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


    def bag_word(self, words, Y=None):
        """ m is the length of dataset
            n is length of wordset
            time complexity: m * length(word term)
        """
        vocabulary_dict = defaultdict(dict)
        for i, word_list in words.iteritems():
            for word in word_list:
                if word not in vocabulary_dict:
                    vocabulary_dict[word] = len(vocabulary_dict)

        word_set_len = len(vocabulary_dict)
        bag_words = []

        for i, word_list in words.iteritems():
            one_bag = [0] * word_set_len
            for word in word_list:
                one_bag[vocabulary_dict[word]] += 1
            if Y:
                one_bag.extend(Y[i])
            bag_words.append(one_bag)
        return bag_words, vocabulary_dict


    def bag_word_vec(self, words, Y=None):
        """ m is the length of dataset
            n is length of wordset
            time complexity: m * n * length(word term)
        """
        word_set = set()
        for i, word_list in words.iteritems():
            word_set.update(word_list)

        word_set = list(word_set)
        for i, word_list in words.iteritems():
            # term count
            one_bag = [word_list.count(word) for word in word_set]
            if Y:
                one_bag.extend(Y[i])
            yield one_bag


    def term_frequency(self, bag_words):
        """ https://en.wikipedia.org/wiki/Tf%E2%80%93idf  Variants of TF weight

            对每一个词频向量进行比例缩放，使它们的每一个元素都在0到1之间，并且不会丢失太多有价值的信息。
            确保每个向量的L2范数等于1，一个计数为1的词在一个向量中的值和其在另一个向量中的值不再相同。
            如果想让一个文档看起来和一个特定主题更相关，你可能会通过不断重复同一个词，来增加它包含一个主题的可能性。在某种程度上，我们得到了一个在该词的信息价值上衰减的结果。所以我们需要按比例缩小那些在一篇文档中频繁出现的单词的值。

        """
        import numpy as np
        import math
        tf_bag_words = []
        for one_bag in bag_words:
            norm = math.sqrt( np.sum([word**2 for word in one_bag]) )
            tf_bag_words.append( [word / norm for word in one_bag] )
        return tf_bag_words

    def inverse_doc_frequency(self, bag_words, vocabulary_dict):
        for word, idx in vocabulary_dict.iteritems():
            a = [1 if word in one_bag else 0 for one_bag in bag_words]
            suma = np.sum(a)





class DataGen(object):
    def __init__(self, filename=None):
        self.wordvec = WordVec(filename)

    def generator_data(self, batch_num=10):
        words, Y = self.wordvec.load_words()
        one_bag = self.wordvec.bag_word_vec(words, Y)

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

    def explore_words(self):
        all_words = []
        words, Y = self.wordvec.load_words()
        for i, word_list in words.iteritems():
            for word in word_list:
                all_words.append(word)

        import pandas as pd
        df = pd.DataFrame(all_words)
        df.columns = ['word']
        top_50_words = df.word.value_counts()[:50]
        return top_50_words


if __name__ == '__main__':
    obj = DataGen('../data/weibo_train_participle.txt')
    bags = obj.generator_data()
    bags.next()

