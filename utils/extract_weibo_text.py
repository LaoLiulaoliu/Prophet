#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os.path
import sys


def get_text(filename):
    content = []
    with open(filename) as fd:
        for line in fd:
            text = '\\'.join( line.split('\t')[6:] )
            content.append(text)
    return content

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    if len(sys.argv) < 3:
        print globals()['__doc__'] % locals()
        sys.exit(1)
    inp, outp = sys.argv[1:3]

    
    output = open(outp, 'w')
    i=0
    for text in get_text(inp):
        output.write(text)
        i+=1
        if (i % 10000 == 0):
            logger.info("Saved " + str(i) + " articles")

    output.close()
    logger.info("Finished Saved " + str(i) + " articles")

