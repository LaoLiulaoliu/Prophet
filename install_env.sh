#!/bin/bash


sudo apt-get install libatlas-base-dev gfortran

Bin="env/bin"

if ! test -d "$Bin" ; then
	virtualenv env
fi

${Bin}/pip install Cython
${Bin}/pip install scipy==0.15.1
${Bin}/pip install gensim
