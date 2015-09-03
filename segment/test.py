#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuande Liu <miraclecome (at) gmail.com>

import time
import subprocess

NUM = 10

def str_process():
    """ use 4.1656 seconds to process one weibo record in NlpAnalysis way
    """
    cmd = 'java -jar target/segment-1.0-SNAPSHOT.jar '
    content = get_text('../weibo_train_data.txt')

    begin = time.time()
    count = 0
    for sentence in content:
        p = subprocess.Popen(cmd + " -s " + sentence.replace(' ', '\\'),
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT)
        stdout, stderr = p.communicate()
        count += 1
        if count >= NUM: break

    end = time.time()
    print(end-begin)

def file_process():
    """ process 8428.77 weibo record every second in NlpAnalysis way
    """
    cmd = 'java -jar target/segment-1.0-SNAPSHOT.jar '

    begin = time.time()
    p = subprocess.Popen(cmd + " -f weibo_text.txt",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)
    stdout, stderr = p.communicate()

    end = time.time()
    print(end-begin)


def get_text(filename):
    content = []
    with open(filename) as fd:
        for line in fd:
            text = '\\'.join( line.split('\t')[6:] )
            content.append(text)
    return content



if __name__ == '__main__':
    str_process()
#    file_process()

