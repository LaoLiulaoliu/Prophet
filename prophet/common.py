# vim: set fileencoding=utf-8 :
#!/usr/bin/env python

import numpy as np
import theano.tensor as T
import keras
from theano.ifelse import ifelse
import theano
import prophet.metric as me
import time

thre = np.array([5.0,3.0,3.0], dtype='float32')
weight_dev = np.array([0.5,0.25,0.25], dtype='float32')

def weibo_loss(y_true, y_pred):
    return ((T.abs_(y_pred - y_true)/(y_true+thre))**2).sum(-1)

def weibo_loss_weighted(y_true, y_pred):
  return (((T.abs_(y_pred - y_true)/(y_true+thre))**2)*weight_dev).sum(-1)

def weibo_loss_scaled_weighted(y_true, y_pred):
  return (((T.abs_(y_pred - y_true)/(y_true+thre) * 10)**2)*weight_dev).sum(-1)

# looks like very slow.
def build_percision_funcs(y_train, y):
  thre = np.array([5.0,3.0,3.0], dtype='float32')
  weight_dev = np.array([0.5,0.25,0.25], dtype='float32')
  #y_gt = T.vector('gt', dtype='float32')
  #y_pre = T.vector('pre', dtype='float32')
  #gt = T.scalar('gt')
  #pre = T.scalar('pre')
  n_total = lambda y_gt: y_gt.sum()
  n_count = lambda y_gt: ifelse(T.le(n_total(y_gt), 100.0), n_total(y_gt), 100.0)
  dev = lambda y_pre, y_gt: ((T.abs_(y_pre - y_gt)/(y_gt + thre))*weight_dev).sum()
  sgn = lambda y_pre, y_gt: ifelse(T.le(1-dev(y_pre, y_gt)-0.8, 0.0), 0.0, 1.0)
  top_pre = lambda y_pre, y_gt: (n_count(y_gt) + 1) * sgn(y_pre, y_gt)
  bt_pre = lambda y_gt: (n_count(y_gt) + 1)
  train_top_precision_components, train_top_precision_updates = theano.scan(fn=top_pre, outputs_info=None, sequences=[y_train, y]) #, none_sequences=[y_gt, y_pre, thre, weight_dev])
  train_top_precision_total = train_top_precision_components.sum()
  #train_top_precision = theano.function(inputs=[self.y_train, self.y], outputs=train_top_precision_total)
  #train_top_precision = train_top_precision(self.y_train, self.y)
  train_bt_precision_components, train_bt_precision_updates = theano.scan(fn=bt_pre, outputs_info=None, sequences=[y])
  train_bt_precision_total = train_bt_precision_components.sum()
  return [train_top_precision_total, train_bt_precision_total]

def build_precisio_stack(y_train, y):
  return [y_train, y]


class WeiboPrecisionCallback(keras.callbacks.Callback):
  def __init__(self, n_epoch_training = 10, n_epoch_test = 5):
    super(WeiboPrecisionCallback, self).__init__()
    self._top_count = 0
    self._bottom_count = 0
    self._y_pred = []
    self._y = []
    self._n_epoch_training = n_epoch_training
    self._n_epoch_test = n_epoch_test
     
  def on_epoch_begin(self, epoch, logs={}):
    self._top_count = 0
    self._bottom_count = 0
    self._y = []
    self._y_pred = []
     
  def on_epoch_end(self, epoch, logs={}):
    #precision = self._top_count / self._bottom_count
    # update the precision.
    #print("On %d epoch, the precision is: %.4f" % (epoch, precision))
    is_val_on = False
    if 'val_more_func_0' in logs and epoch % self._n_epoch_test == 0:
      is_val_on = True
    is_train_on = False
    if epoch % self._n_epoch_training == 0:
      is_train_on = True

    if (not is_train_on) and (not is_val_on):
      return
      
    print("Calculating the weibo precision...")
    traing_msg=""
    if is_train_on:
      start=time.time()
      precision = me.WeiboPrecision.precision_match(self._y, self._y_pred) 
      end=time.time()
    
    if is_val_on:
      traing_msg="training"
      start_val = time.time()
      #print(logs['val_more_func_1'], logs['val_more_func_0'])
      precision_val = me.WeiboPrecision.precision_match(logs['val_more_func_1'], np.rint(logs['val_more_func_0']))
      precision_val_non_round = me.WeiboPrecision.precision_match(logs['val_more_func_1'], logs['val_more_func_0'])
      end_val = time.time()
      print("On %d epoch, the %s precision is: %.4f, non-round: %.4f, calcuating time: %f" % (epoch, "validation", precision_val, precision_val_non_round, end_val-start_val))
      #f=theano.function([], weibo_loss_scaled_weighted(np.array(logs['val_more_func_1']), np.array(logs['val_more_func_0'])))
      #print("loss", f().mean())

    if not is_train_on:
      return  
    print("On %d epoch, the %s precision is: %.4f, calcuating time: %f" % (epoch, traing_msg, precision, end-start))
    
    
     
  def on_batch_begin(self, batch, logs={}):
    pass
     
  def on_batch_end(self, batch, logs={}):
    #top: more_func_0 bottom: more_func_1
    #self._top_count += float(logs['more_func_0'])
    #self._bottom_count += float(logs['more_func_0'])
    self._y_pred.extend(logs['more_func_0'])
    self._y.extend(logs['more_func_1'])
    