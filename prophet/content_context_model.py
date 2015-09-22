# vim: set fileencoding=utf-8 :
#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import theano
from six.moves import cPickle
import os, re, json

from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Flatten, Dropout
from keras.models import Sequential 
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils, generic_utils
from keras.regularizers import l2
from prophet.weibo_reader import WeiboReader
from prophet.metric import WeiboPrecision
from prophet.common import weibo_loss, weibo_loss_weighted, weibo_loss_scaled_weighted, WeiboPrecisionCallback, build_precisio_stack
from keras.callbacks import ModelCheckpoint
from prophet.ppl_idx_table import PplIdxTable
from keras.layers.recurrent import LSTM
from keras.preprocessing import sequence

from gensim.models import Word2Vec
from prophet.data import WeiboDataset

dim=100

def collect_words_vec(word_model, words, max_len):
  vec = np.zeros([max_len, dim], dtype='float32')
  offset = 0
  for idx, word in enumerate(words):
    if word in word_model:
      if idx - offset >= max_len:
        return vec
      vec[idx-offset,:] = word_model[word]
    else:
      offset += 1      
  return vec

def find_max_seq(data, max_len):
  ret = max_len
  for info in data:
    if len(info) > 7:
      size = len(info[7])
    else:
      size = len(info[4])
    if size > 120:
      print(info[0], info[1])
      if len(info) > 7:
        for inf in info[7]:
          print(inf)
      else:
          for inf in info[4]:
            print(inf)
    if ret < size:
      ret = size
  return ret
    
save_dir = "gen_nn_model"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    

dataset = WeiboDataset()
print('Loading the data...')
dataset.load_data(
             ("./data/weibo_train_data.txt", "./gen_data/weibo_train_data_text.txt.jian.words"),
             ("./data/weibo_predict_data.txt", "./gen_data/weibo_predict_data_text.txt.jian.words")
             )
print('Loading word vector model')
word_vec_filename="./gen_model/vec_state_s100_p0_w5_t1.all"
dataset.load_words_vec(word_vec_filename, max_len=20)


train_gt = dataset.get_training_data_gt_np()
val_gt = dataset.get_validation_data_gt_np()
print("The max len of words is: ", dataset.get_missing_info(is_valid=False, is_predict=False, is_max_len=True))

train_words = dataset.get_words_vec_training_data_np()
val_words = dataset.get_words_vec_validation_data_np()
predict_words = dataset.get_words_vec_predict_data_np()

print('Building the model')
model = Sequential()
model.add(LSTM(100, 512, return_sequences=True))  # try using a GRU instead, for fun
model.add(Dropout(0.5))
model.add(LSTM(512, 512, return_sequences=False))
model.add(Dropout(0.6))
model.add(Dense(512, 1024, init='uniform', activation="linear",
                W_regularizer=l2(0.01)))
model.add(Dense(1024, 3, init="uniform", activation="linear",
                W_regularizer=l2(0.01)))
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss=weibo_loss_scaled_weighted, optimizer=sgd, other_func_init=build_precisio_stack)
#model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

print("Traing the model")
checkpoint = ModelCheckpoint(save_dir+"/content_context_state.full_t.pkl", save_best_only=False)
#print(train_words[0])
print(train_words[0][0])
precision = WeiboPrecisionCallback()
model.fit(train_words, train_gt, batch_size=256, nb_epoch=121, show_accuracy=True, callbacks=[checkpoint, precision], validation_data=(val_words, val_gt))

print("predict shape: ", predict_words.shape)
pre=model.predict(predict_words, batch_size=128)
dataset.save_predictions(pre, './ppl_result.txt')
exit(1)

