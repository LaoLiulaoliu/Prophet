#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
import logging
import os.path
import sys
import multiprocessing
import argparse
 
from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models import Doc2Vec
from gensim.models import Phrases
from gensim.models.word2vec import LineSentence

import string
import multiprocessing

vec_size=200
window_size=5
min_count=5
workers=multiprocessing.cpu_count()
phrases=0
is_document=False
phrases_list=[]
g_sg=1

def get_phrase_list(p_list, n, sen):
  if n == 0:
    return sen
  else:
    return p_list[n-1][get_phrase_list(p_list, n-1, sen)]

def load_words(filename, words):
    Y = []
    with open(filename) as fd:
        for line in fd:
            start = 0
            if line[0] == '[':
                start=1
            end = -1
            if line[-1] == ']':
                end = -1
            if line[-2] == ']' and line[-1] == '\n':
                end = -2
            if line[-1] == '\t' and line[-2] == '\r' and line[-3] == ']':
                end = -3
            sentence = filter(lambda x: x != '', map(string.strip, line[start:end].split(',')))
            words.append(sentence)
    return words

 
if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
 
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    parser = argparse.ArgumentParser("Train word2vec")
    parser.add_argument('-o', '--output', type=str, help='output model name')
    parser.add_argument('-d', '--document', default=False, action='store_true', help='we will train with doc2vec')
    parser.add_argument('-p', '--phrase', type=int, help='enable finding phrase procedure')
    parser.add_argument('-s', '--size', type=int, help='the vector size')
    parser.add_argument('-w', '--window', type=int, help='the window size')
    parser.add_argument('-m', '--min_count', type=int, help='The min_count size')
    parser.add_argument('-k', '--workers', type=int, help='the number of workders')
    parser.add_argument('-t', '--t_algo', type=int, help='the training algorithm, 1: skip-gram')
    parser.add_argument('files', nargs='+', help='the sentence files')

    args = parser.parse_args()

    if args.size:
        vec_size=int(args.size)
    if args.window:
        window_size=int(args.window)
    if args.min_count:
        min_count=int(args.min_count)
    if args.workers:
        workers=int(args.workers)
    if args.phrase:
        phrases=int(args.phrase)
    if args.document:
        is_document=args.document
    if args.t_algo:
        g_sg=int(args.t_algo)


    sentences=[]
    if args.files:
        logger.info("parsing " + str(len(args.files)) + " files")
        for filename in args.files:
            load_words(filename, sentences)
    
    logger.info("Total %d sentences" % len(sentences))

    if phrases > 0:
        logger.info("Train phrases")
        for i in range(0, phrases):
            phrases_list.append(Phrases(get_phrase_list(phrases_list, i, sentences)))
        logger.info("Saving phrases")
        for idx, ph in enumerate(phrases_list):
            ph.save(args.output+".phrase"+str(idx))
        logger.info("Replace sentences with phrases")
        sentences = get_phrase_list(phrases_list, phrases-1, sentences)

    if is_document:
        logger.info("Train doc2vec")
        model = Doc2Vec(sentences, size=vec_size, window=window_size, 
                        min_count=min_count, workers=workers, dm=g_sg)
    else:
        logger.info("Train word2vec")
        model = Word2Vec(sentences, size=vec_size, window=window_size, 
                         min_count=min_count,workers=workers, sg=g_sg)

    if args.output:
        model.save(args.output)
 
