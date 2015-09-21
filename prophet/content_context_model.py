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
from prophet.common import weibo_loss, weibo_loss_weighted, weibo_loss_scaled_weighted
from keras.callbacks import ModelCheckpoint
from prophet.ppl_idx_table import PplIdxTable
from keras.layers.recurrent import LSTM
from keras.preprocessing import sequence

from gensim.models import Word2Vec

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
    
print('Loading word vector model')
word_vec_filename="./gen_model/vec_state_s100_p0_w5_t1.all"
word_model = Word2Vec.load(word_vec_filename)

max_word_len = 0
print('Loading the data...')
reader = WeiboReader()
reader.load_words_data("./data/weibo_train_data.txt", "./gen_data/weibo_train_data_text.txt.jian.words")
train_data = reader.get_training_data()
train_gt = np.array([[info[3], info[4], info[5]] for info in train_data], dtype='float32')
max_word_len=find_max_seq(reader.data(), max_word_len)
max_word_len=20
#for info in train_data:
#  for word in info[7]:
#    print(word)
#    print(word_model[word])
  
#train_words = np.array([collect_words_vec(word_model, info[7]) for info in train_data ], dtype='float32')

#train_words = pad_sequences_3d(train_words, maxlen=None, dtype='float32', padding="post", truncating="post")

val_data = reader.get_validation_data()
val_gt = np.array([[info[3], info[4], info[5]] for info in val_data], dtype='float32')
#val_words = [word_model[info[7]] for info in train_data]
#val_words = np.array([collect_words_vec(word_model, info[7]) for info in val_data ], dtype='float32')

#val_words = pad_sequences_3d(val_words, maxlen=None, dtype='float32', padding="post", truncating="post")


reader_pre = WeiboReader()
reader_pre.load_words_data("./data/weibo_predict_data.txt", "weibo_predict_data_text.txt.jian.words")
predict_data = reader_pre.data()
max_word_len=find_max_seq(predict_data, max_word_len)
print("The max len of words is: ", max_word_len)
#predict_words = [word_model[info[4]] for info in predict_data]
#predict_words = np.array([collect_words_vec(word_model, info[4]) for info in predict_data], dtype='float32')
#exit(1)
#predict_words = pad_sequences_3d(predict_words, maxlen=None, dtype='float32', padding="post", truncating="post")
train_words = np.array([collect_words_vec(word_model, info[7], max_word_len) for info in train_data ], dtype='float32')
val_words = np.array([collect_words_vec(word_model, info[7], max_word_len) for info in val_data ], dtype='float32')
predict_words = np.array([collect_words_vec(word_model, info[4], max_word_len) for info in predict_data], dtype='float32')
#print("Training words 0 shape", train_words[0].shape)
#print("validation words 0 shape", val_words[0].shape)

print('Building the model')
model = Sequential()
model.add(LSTM(100, 1024, return_sequences=True))  # try using a GRU instead, for fun
model.add(Dropout(0.5))
model.add(LSTM(1024, 2048, return_sequences=False))
model.add(Dropout(0.6))
model.add(Dense(2048, 2048, init='uniform', activation="tanh",
                W_regularizer=l2(0.01)))
model.add(Dense(2048, 3, init="uniform", activation="linear",
                W_regularizer=l2(0.01)))
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss=weibo_loss_scaled_weighted, optimizer=sgd)
#model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

print("Traing the model")
checkpoint = ModelCheckpoint(save_dir+"/content_context_state.full_t.pkl", save_best_only=False)
#print(train_words[0])
print(train_words[0][0])
model.fit(train_words, train_gt, batch_size=256, nb_epoch=120, show_accuracy=True, callbacks=[checkpoint])
train_pre=model.predict(train_words, batch_size=128)
print("traing weibo acc: ", WeiboPrecision.precision_match(train_gt, train_pre))

print("validation acc: ", model.evaluate(val_words, val_gt, 128, show_accuracy=True))
pre=model.predict(val_words, batch_size=128)
print("validation weibo acc: ", WeiboPrecision.precision_match(val_gt, pre))
print("predict shape: ", predict_words.shape)
pre=model.predict(predict_words, batch_size=128)
reader_pre.save_data(pre, './ppl_result.txt')
exit(1)

