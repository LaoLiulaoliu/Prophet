#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os.path
import sys
import time
import subprocess


def extract_words(filename):
    cmd = 'java -jar segment/target/segment-1.0-SNAPSHOT.jar '

    begin = time.time()
    p = subprocess.Popen(cmd + " -f " + filename,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)
    stdout, stderr = p.communicate()

    end = time.time()
    print(end-begin)


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    if len(sys.argv) < 2:
        print globals()['__doc__'] % locals()
        sys.exit(1)

    extract_words(sys.argv[1])

    #logger.info("Finished Saved " + str(i) + " articles")

