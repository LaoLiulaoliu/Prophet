#!/bin/bash


#sudo apt-get install libatlas-base-dev gfortran

Bin="`pwd`/env/bin"

if ! test -d "$Bin" ; then
	virtualenv -p /usr/bin/python env
fi

${Bin}/pip install --upgrade pip
${Bin}/pip install Cython
${Bin}/pip install scipy==0.15.1
${Bin}/pip install h5py
${Bin}/pip install gensim


if ! test -d "library" ; then
	git clone https://github.com/NLPchina/ansj_seg.git
	cp -r ansj_seg/library ./
fi


if ! test -d "lib" ; then
	mkdir lib
fi

if ! test -d "lib/keras" ; then
	(cd lib; git clone https://github.com/dxj19831029/keras)
fi
if test -d "lib/keras" ; then
        (cd lib/keras; ${Bin}/python setup.py install)
	#${Bin}/pip install keras
fi
