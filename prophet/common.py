# vim: set fileencoding=utf-8 :
#!/usr/bin/env python

import numpy as np
import theano.tensor as T
import keras

thre = np.array([5.0,3.0,3.0], dtype='float32')
weight_dev = np.array([0.5,0.25,0.25], dtype='float32')

def weibo_loss(y_true, y_pred):
    return ((T.abs_(y_pred - y_true)/(y_true+thre))**2).sum(-1)

def weibo_loss_weighted(y_true, y_pred):
  return (((T.abs_(y_pred - y_true)/(y_true+thre))**2)*weight_dev).sum(-1)

def weibo_loss_scaled_weighted(y_true, y_pred):
  return (((T.abs_(y_pred - y_true)/(y_true+thre) * 10)**2)*weight_dev).sum(-1)

# class WeiboPrecisionCallback(keras.callbacks.Callback):
#   def __init__(self):
#     super(WeiboPrecisionCallback, self).__init__()
#     self._top_count = 0
#     self._bottom_count = 0
#     
#   def on_epoch_begin(self, epoch, logs={}):
#     keras.callbacks.Callback.on_epoch_begin(self, epoch, logs=logs)
#     print(logs)
#     
#   def on_epoch_end(self, epoch, logs={}):
#     keras.callbacks.Callback.on_epoch_end(self, epoch, logs=logs)
#     print(logs)
#     
#   def on_batch_begin(self, batch, logs={}):
#     keras.callbacks.Callback.on_batch_begin(self, batch, logs=logs)
#     print(logs)
#     
#   def on_batch_end(self, batch, logs={}):
#     keras.callbacks.Callback.on_batch_end(self, batch, logs=logs)
#     print(logs)