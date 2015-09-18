#!/bin/bash


#sudo apt-get install libatlas-base-dev gfortran

Bin="env/bin"

if ! test -d "$Bin" ; then
	virtualenv -p /usr/bin/python env
fi

${Bin}/pip install --upgrade pip
${Bin}/pip install Cython
${Bin}/pip install scipy==0.15.1
${Bin}/pip install gensim


if ! test -d "library" ; then
	git clone https://github.com/NLPchina/ansj_seg.git
	cp -r ansj_seg/library ./
fi

