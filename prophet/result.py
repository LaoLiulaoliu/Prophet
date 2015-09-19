#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuande Liu <miraclecome (at) gmail.com>

from __future__ import print_function, division

count = '0,0,0'

with open('result.txt', 'w') as fd_re:
    with open('../data/weibo_predict_data.txt') as fd:
        for line in fd:
            uid, mid, _ = line.split('\t', 2)
            fd_re.write('{}\t{}\t{}\n'.format(uid, mid, count))

