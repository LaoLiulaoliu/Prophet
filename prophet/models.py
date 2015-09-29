# vim: set fileencoding=utf-8 :
#!/usr/bin/env python

from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Flatten, Dropout, Merge, Activation
from keras.models import Sequential 
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.regularizers import l2
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import *
import os
import theano
import theano.tensor as T
from keras.constraints import *

def weibo_act_zero(x):
  return T.switch(T.lt(x, 100000), T.switch(T.lt(x, 0), 0, x), 100000)

def build_ppl_context_model(max_ppl, dim_proj=300, dim_output=3, saved_filename = None, is_output=True):
  model = Sequential()
  model.add(Embedding(max_ppl, dim_proj, init="uniform"))
  model.add(Dropout(0.7))
  model.add(Flatten())
  if is_output:
    model.add(Dense(dim_proj, dim_output, init="uniform", activation=weibo_act_zero,
                  W_regularizer=l2(0.01), W_constraint = maxnorm(10)))
  #else:
  #  model.add(Dense(dim_proj, 1024, init="uniform", activation="linear",
  #                W_regularizer=l2(0.01)))
  if saved_filename is not None and os.path.isfile(saved_filename):
    model.load_weights(saved_filename)
  return model


def build_content_context_model_2_lstm(in_dim=100, hidden1=512, hidden2=512, 
        dense1=1024, dim_output=3, drop1=0.5, drop2=0.6, 
        saved_filename = None, is_output=True):
  model = Sequential()
  model.add(LSTM(in_dim, hidden1, return_sequences=True))  # try using a GRU instead, for fun
  model.add(Dropout(drop1))
  model.add(LSTM(hidden1, hidden2, return_sequences=False))
  model.add(Dropout(drop2))
  model.add(Dense(hidden2, dense1, init='uniform', activation="linear",
                  W_regularizer=l2(0.01)))
  if is_output:
    model.add(Dense(dense1, dim_output, init="uniform", activation="linear",
                  W_regularizer=l2(0.01)))
  if saved_filename is not None and os.path.isfile(saved_filename):
    model.load_weights(saved_filename)
  return model

def build_content_contex_model_lstm(in_dim=100, hidden1=512, dense1=1024, 
                                    dim_output=3, drop1=0.5,
                                    saved_filename=None, is_output=True):
  model = Sequential()
  model.add(LSTM(in_dim, hidden1, return_sequences=False))  # try using a GRU instead, for fun
  model.add(Dropout(drop1))
  model.add(Dense(hidden1, dense1, init='uniform', activation="linear",
                  W_regularizer=l2(0.01)))
  if is_output:
    model.add(Dense(dense1, dim_output, init="uniform", activation="linear",
                  W_regularizer=l2(0.01)))
  if saved_filename is not None and os.path.isfile(saved_filename):
    model.load_weights(saved_filename)
  return model

def weibo_act(x):
  return T.switch(T.lt(x, 100000), T.abs_(x), 100000)



def build_combine_model(max_ppl, dim_proj=300, vec_dim=100, 
                        lstm_hidden=512, dim_output=3, 
                        saved_filename=None, is_output=True):
  #ppl_model = build_ppl_context_model(max_ppl, dim_proj, 
  #                          dim_output, is_output=False)
  ppl_model = Sequential()
  ppl_model.add(Embedding(max_ppl, dim_proj, init="uniform"))
  #model.add(Dropout(0.7))
  ppl_model.add(Flatten())
  
  dense1=2048
  content_model = Sequential()
  content_model.add(LSTM(vec_dim, lstm_hidden, return_sequences=True))  # try using a GRU instead, for fun
  content_model.add(Dropout(0.2))
  content_model.add(LSTM(lstm_hidden, lstm_hidden, return_sequences=False))  # try using a GRU instead, for fun
  content_model.add(Dropout(0.5))
  content_model.add(Dense(lstm_hidden, dense1, init='uniform', activation="tanh",
                  W_regularizer=l2(0.01), b_regularizer=l2(0.01)))
  content_model.add(Dense(dense1, dense1, init='uniform', activation="tanh",
                  W_regularizer=l2(0.01), b_regularizer=l2(0.01)))
  
     
  #content_model = build_content_contex_model_lstm(vec_dim, 
  #                            lstm_hidden, 
  #                            dim_output=dim_output, 
  #                            dense1=dense1,
  #                            drop1=0.5, 
  #                            saved_filename=None, 
  #                            is_output=False)
  model=Sequential()
  model.add(Merge([ppl_model, content_model], mode='concat', concat_axis=1))
  model.add(Dropout(0.6))
  model.add(Dense(dim_proj+dense1, 1024, activation='tanh',
                  W_regularizer=l2(0.01), b_regularizer=l2(0.01)))
  model.add(Dropout(0.5))
  model.add(Dense(1024, 512, activation='tanh',
                  W_regularizer=l2(0.01), b_regularizer=l2(0.01)))
  model.add(Dropout(0.8))
  if is_output:
    model.add(Dense(512, dim_output, init="uniform", activation=weibo_act,
                  W_regularizer=l2(0.01), b_regularizer=l2(0.01), W_constraint = maxnorm(2)) )

  if saved_filename is not None and os.path.isfile(saved_filename):
    model.load_weights(saved_filename)
        
  return model

def sharedX(X, dtype=theano.config.floatX, name=None):
    return theano.shared(np.asarray(X, dtype=dtype), name=name)
  
def normal_init(shape):
    return sharedX((2*np.random.randn(*shape)-1))
  
def build_conv2d_model(nb_feat1=32, 
                       words = 30, words_vec = 100, output_dim=3,
                       saved_filename=None, is_output=True):
#   conv_model = Sequential()
#   conv_model.add(Convolution2D(nb_feat1, 1, nb_row, nb_col, 
#                                init="normal" ,border_mode="full"))
#   conv_model.add(Activation('relu'))
#   conv_model.add(Convolution2D(nb_feat1, nb_feat1, nb_row, nb_col, 
#                                init="normal" ,border_mode="full"))
#   conv_model.add(Activation('relu'))
#   conv_model.add(MaxPooling2D(poolsize=(nb_pool,nb_pool)))
#   conv_model.add(Dropout(0.25))
#   conv_model.add(Flatten())
#   
#   # (the number of filters is determined by the last Conv2D)
#   conv_model.add(Dense(nb_feat1 * (words / nb_pool) * (words_vec / nb_pool), 
#                        2048, activation="relu"))
#   conv_model.add(Dropout(0.5))
  n_gram = 3
  w_decay = 0.0005
  conv_model = Sequential()
  conv_model.add(Convolution2D(nb_feat1, 1, n_gram, words_vec, init="normal", subsample=(1,words_vec),
                  W_regularizer=l2(w_decay), W_constraint = maxnorm(10)))
  conv_model.add(Activation('relu'))
  pool_size = 4
  stride_size = 2
  conv_model.add(MaxPooling2D(poolsize=(pool_size, 1), stride=(stride_size, 1)))
  #conv_model.add(Dropout(0.5))
  conv_model.add(Flatten())
  #dense1 = nb_feat1 #words - n_gram + 1
  conv_size = words - n_gram + 1
  dense1 = nb_feat1 * ((conv_size - pool_size) / stride_size + 1)
  print("dense1: ", dense1)
  conv_model.add(Dense(dense1, 2048, activation="relu",
                  W_regularizer=l2(w_decay), W_constraint = maxnorm(10) ))
  conv_model.add(Dropout(0.5))
  conv_model.add(Dense(2048, 2048, activation="relu",
                  W_regularizer=l2(w_decay), W_constraint = maxnorm(10) ))
  conv_model.add(Dropout(0.8))
  if is_output:
    conv_model.add(Dense(2048, output_dim, activation=weibo_act, init=normal_init,
                  W_regularizer=l2(w_decay), W_constraint = maxnorm(1000)))
  
  if saved_filename is not None and os.path.isfile(saved_filename):
    conv_model.load_weights(saved_filename)

  return conv_model

