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

class RankedWeiboLoss():
  def __init__(self, dataset):
    self._f, self._c, self._l = dataset.get_ranked_weighted_metric_np()
    
  def calculate_loss(self, y_true, y_pred):
    y_true_int = T.cast(y_true, dtype='int32')
    s0 = T.constant(self._f, 'f')[y_true_int[:,0]]
    s1 = T.constant(self._c, 'c')[y_true_int[:,1]]
    s2 = T.constant(self._l, 'l')[y_true_int[:,2]]
    total = s0+s1+s2
    return T.switch(total > 100, 101, total+1)*(((y_pred-y_true)**2).sum(-1))

def weibo_loss(y_true, y_pred):
    return ((T.abs_(y_pred - y_true)/(y_true+thre))**2).sum(-1)

def weibo_loss_weighted(y_true, y_pred):
  return (((T.abs_(y_pred - y_true)/(y_true+thre))**2)*weight_dev).sum(-1)

def weibo_loss_scaled_weighted(y_true, y_pred):
  return (((T.abs_(y_pred - y_true)/(y_true+thre) * 10)**2)*weight_dev).sum(-1)

def weibo_precision_loss(y_true, y_pred):
  devs = T.abs_(y_pred - y_true) / (y_true + thre) * weight_dev
  total_devs = devs.sum(-1)
  
  loss_factor = T.switch(y_true.sum(-1) > 100.0, y_true.sum(-1)*0.0 + 100.0, y_true.sum(-1)+1) 
  loss = (total_devs ** 2) * (loss_factor)
  switch_loss = T.switch(total_devs > 0.2, loss, loss*0)
  return switch_loss

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

def build_precisio_stack_softmax(y_train, y):
  return [T.argmax(y_train[0], -1), T.argmax(y[0], -1)]

def rank_limit(limit=1000, alpha=0.2, beta=5):
  """
    let's say allow error rate is 0.2 for each prediction.
    alpha = abs(pre - gt) / (gt + beta)
    a is lower ground truth
    b is upper ground truth
    c is the prediction
    (c - a) / (a + beta) = alpha
    (b - c) / (b + beta) = alpha
    c = (1+alpha)*a + alpha*beta
    b = (c + alpha*beta) / 0.8
    Note: alpha will be minus 0.0001 (very small number, since 0.2 is not valid) 
    
    we will loop from smallest number to largest
    Returns:
      [(a,c,b)]
  """
  alpha -= 0.0001
  last_b = -1
  limit_list = []
  for a in range(0, limit):
    if a <= last_b:
      continue
    c = int((1+alpha)*a+alpha*beta) 
    b = int((c+alpha*beta)/(1-alpha))
    limit_list.append((a,c,b))
    last_b = b
  return limit_list

def to_categorical(gt, classes):
  Y = np.zeros(gt.shape + (classes,))
  for row in range(gt.shape[0]):
    for col in range(gt.shape[1]):
      i = gt[row,col]
      Y[row,col,i] = 1
  return Y


class WeiboPrecisionCallback(keras.callbacks.Callback):
  def __init__(self, n_epoch_training = 10, n_epoch_test = 5, val_saved_filename=None, dataset_ranking = None):
    super(WeiboPrecisionCallback, self).__init__()
    self._top_count = 0
    self._bottom_count = 0
    self._y_pred = []
    self._y = []
    self._n_epoch_training = n_epoch_training
    self._n_epoch_test = n_epoch_test
    self._val_saved_filename = val_saved_filename
    self._dataset_ranking = dataset_ranking
     
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
      if self._dataset_ranking is not None:
        y_pred = self._dataset_ranking.translate_ranking(np.int16(np.rint(self._y_pred)))
        y = self._dataset_ranking.get_training_data_gt_np()
      else:
        y_pred = self._y_pred
        y = self._y
      precision = me.WeiboPrecision.precision_match(y, y_pred) 
      end=time.time()
    
    if is_val_on:
      traing_msg="training"
      start_val = time.time()
      #print(logs['val_more_func_1'], logs['val_more_func_0'])
      if self._dataset_ranking is not None:
        y_pred = self._dataset_ranking.translate_ranking(np.int16(np.rint(logs['val_more_func_0'])))
        #y = logs['val_more_func_1']
        y = self._dataset_ranking.get_validation_data_gt_np()
        precision_val = me.WeiboPrecision.precision_match(y, y_pred)
        precision_val_non_round = precision_val
        
      else:
        y_pred = np.rint(logs['val_more_func_0'])
        y = logs['val_more_func_1']
        precision_val = me.WeiboPrecision.precision_match(y, y_pred)
        precision_val_non_round = me.WeiboPrecision.precision_match(y, logs['val_more_func_0'])
      end_val = time.time()
      if self._val_saved_filename is not None:
        fd = open(self._val_saved_filename, 'w')
        for gt, pre, pre_f in zip(y, y_pred, logs['val_more_func_0']):
          fd.write('%d,%d,%d-%d,%d,%d-%f,%f,%f\n'%(gt[0], gt[1], gt[2], pre[0], pre[1], pre[2], pre_f[0], pre_f[1], pre_f[2]))
        fd.close()
        print("Saved %d valiation results to %s" % (len(logs['val_more_func_0']), self._val_saved_filename))
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
    
def detect_nan(i, node, fn):
    for output in fn.outputs:
        if (not isinstance(output[0], np.random.RandomState) and
            np.isnan(output[0]).any()):
            print('*** NaN detected *** i: ', i)
            theano.printing.debugprint(node)
            print(node)
            print("input nan: ", np.isnan(fn.inputs[0][0]).any())
            print('Inputs : %s' % [(input[0], np.asarray(input[0]).shape) for input in fn.inputs])
            #print('numpy result is %s' % ([numpy.asarray(input[0]).tolist() for input in fn.inputs]))
            print('Outputs: %s' % [(output[0], np.asarray(output[0]).shape) for output in fn.outputs])
            #print(np.asarray(fn.inputs[0][0]))
            #print(fn.outputs[0][0].shape)
            #for vec in np.asarray(fn.outputs[0][0]).tolist():  
            #  if np.isnan(np.array(vec)):
            #    print(vec)
            #print(np.asarray(fn.outputs[0][0]).tolist())
            #print(np.asarray(fn.outputs[0][0]).tolist())
              
            #print('numpy result is %s' % ([numpy.asarray(output[0]).tolist() for output in fn.outputs]))
            exit(1)

    