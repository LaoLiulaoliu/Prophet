# vim: set fileencoding=utf-8 :
#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function

import os
prophet_root = '/home/xijing/Work/weibo/Prophet'
import sys
sys.path.insert(0, prophet_root)


from keras.optimizers import SGD, RMSprop, Adagrad
from keras.callbacks import ModelCheckpoint

from prophet.metric import WeiboPrecision
from prophet.common import *

from prophet.data import WeiboDataset
from prophet.models import *
from keras.layers.core import *
from keras.models import Graph
from keras.utils import np_utils

max_features = 1000000  # vocabulary size: top 50,000 most common words in data
skip_top = 0  # ignore top 100 most common words
nb_epoch = 1
dim_proj = 300  # embedding space dimension
dim_output = 3 # the output predicting values.

save = True
is_ranking = True
#load_model = False
#train_model = True
print("Training ppl context")

save_dir = "gen_nn_model"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
model_load_fname = "ppl_context.pkl"
model_save_fname = "ppl_context.pkl"

print('Loading the data...')
import time
start = time.time()
is_ranking=True
data = WeiboDataset(ranking_func=rank_limit)
data.load_data("./data2/weibo_train_data.txt", "./data2/weibo_predict_data.txt", is_init_ppl_standard=True)


train_gt = data.get_training_data_gt_np(is_ranking=is_ranking)
train_ppl = data.get_ppl_training_data_np()
val_gt = data.get_validation_data_gt_np(is_ranking=is_ranking)
val_ppl = data.get_ppl_validation_data_np()

predict_ppl = data.get_ppl_predict_data_np()

missing_info = data.get_missing_info(is_max_len=False)
print("cost time: ", str(time.time() - start))

print("-- validation missing %d users, %d weibo" % (missing_info['val'], missing_info['val_c']))
print("-- predict data missing %d users, %d weibo" % (missing_info['pre'], missing_info['pre_c']))
  
max_features = data.get_ppl_max_count()

print('Build model...')
#model = build_ppl_context_model(max_features, dim_proj, dim_output, is_ranking=is_ranking)
max_norm_v = 29
model = Graph()
#print(train_ppl.shape[1:])
model.add_input(name='input', input_shape = train_ppl.shape[1:], dtype='int32')
model.add_node(Embedding(max_features, dim_proj, init="uniform", input_length=1), name='embedding', input='input')
model.add_node(Dropout(0.7), name='drop1', input='embedding')
#print(model.get_config())
model.add_node(Flatten(), name='flatten1', input='drop1')
max_ranks_len = data.max_ranks()
model.add_node(Dense(max_ranks_len, init="normal", activation='softmax',
                  W_regularizer=l2(0.01), W_constraint = maxnorm(max_norm_v)), name='softmax_f', input='flatten1')
model.add_node(Dense(max_ranks_len, init="normal", activation='softmax',
                  W_regularizer=l2(0.01), W_constraint = maxnorm(max_norm_v)), name='softmax_c', input='flatten1')
model.add_node(Dense(max_ranks_len, init="normal", activation='softmax',
                  W_regularizer=l2(0.01), W_constraint = maxnorm(max_norm_v)), name='softmax_l', input='flatten1')
#model.add_node(Reshape(3, max_ranks_len), inputs=['softmax_f', 'softmax_c', 'softmax_l'], merge_mode='concat', concat_axis=1, name='softmax3', create_output=True)
model.add_node(Reshape(dims=(3, max_ranks_len)), inputs=['softmax_f', 'softmax_c', 'softmax_l'], merge_mode='concat', concat_axis=1, name='softmax3_pre', create_output=False)
model.add_output(input='softmax3_pre', name='softmax3')
#model.add_output(inputs=['softmax_f', 'softmax_c', 'softmax_l'], merge_mode='concat', concat_axis=1, name='softmax3')
#model.add_output(name='softmax3', input='softmax3_pre')
#print(model.get_config())
#print(model.nodes['softmax3_pre'].get_config())
#print(model.outputs['softmax3'].get_config())
#print(model.outputs['softmax3'].get_output(train=True))
#print(model.outputs['softmax3'].get_output(train=False))
#print(model.outputs['softmax3'].output_shape)
#print(model.outputs['softmax3'].get_input(train=True).shape)
#print(model.outputs['softmax3'].get_input(train=False).shape)
#exit(1)



#model.compile(loss=weibo_loss, optimizer='rmsprop')
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
#sgd = SGD(lr=2, decay=1e-6, momentum=0.9, nesterov=True)
#use weibo_loss_weighted with learning rate 2 is good.
#model.compile(loss=weibo_loss_weighted, optimizer=sgd)
if is_ranking:
  obj = RankedWeiboLoss(data)
  loss_func = obj.categorical_crossentropy
else:
  loss_func = weibo_loss_weighted
#model.compile(loss=loss_func, optimizer=sgd, other_func_init=build_precisio_stack)
#model.compile(loss="mse", optimizer=sgd)
#model.compile(sgd, loss={'softmax_f':'binary_crossentropy', 'softmax_c':'binary_crossentropy', 'softmax_l':'binary_crossentropy'})

model.compile(sgd, loss={'softmax3':loss_func}, other_func_init=build_precisio_stack_softmax, weighted_method="multi")
#exit(1)
#model.compile(sgd, loss={'softmax3':'categorical_crossentropy'}, other_func_init=build_precisio_stack_softmax)

checkpoint = ModelCheckpoint(save_dir+"/ppl_context_state2.full_t.10.f10.pkl", save_best_only=False)
if is_ranking:
  precision = WeiboPrecisionCallback(1, 1, dataset_ranking=data, val_saved_filename="./test_saved_ppl_softmax.val")
else:
  precision = WeiboPrecisionCallback(1, 1)
#train_gt = np.reshape(train_gt, train_gt.shape + (1,))
       
train_gt = to_categorical(train_gt, max_ranks_len)
val_gt = to_categorical(val_gt, max_ranks_len)
mat = data.get_ranked_weighted_metric_matrix_np()
#print(mat.shape)
#print(mat.dtype)
#print(val_gt)
#print(val_gt.shape)
#print((val_gt * mat).sum(-1).sum(-1))

#a=T.tensor3('a', dtype='float32')
#b=T.tensor3('b', dtype='float32')
#cce = T.nnet.categorical_crossentropy(a,b)
#f = theano.function([a,b],cce)

#i_d = val_gt
#print(i_d, i_d.shape)
#print(f(i_d, i_d))

#exit(1)
#val_gt = np.reshape(val_gt, val_gt.shape + (1,))
#print(train_ppl.shape)
#print(train_gt.shape)
#model.fit({'input':train_ppl, 'softmax_f':train_gt[:,0], 'softmax_c':train_gt[:,1], 'softmax_l':train_gt[:,2]}, batch_size=256, nb_epoch=120, callbacks=[checkpoint], validation_data={'input':val_ppl, 'softmax_f':val_gt[:,0], 'softmax_c':val_gt[:,1], 'softmax_l':val_gt[:,2]})
print("train gt: ", train_gt.shape, train_gt.dtype)
model.fit({'input':train_ppl, 'softmax3':train_gt}, batch_size=256, nb_epoch=20, callbacks=[checkpoint, precision], validation_data={'input':val_ppl, 'softmax3':val_gt})
#model.fit(train_ppl, train_gt, batch_size=256, nb_epoch=120, show_accuracy=True, callbacks=[checkpoint], validation_data=(val_ppl, val_gt))

pre=model.predict({'input':val_ppl}, batch_size=128)

pre = pre['softmax3'].argmax(-1)

if is_ranking:
  pre = data.translate_ranking(pre)
  val_gt = data.get_validation_data_gt_np()
  
print("validation weibo acc: ", WeiboPrecision.precision_match(val_gt, pre))

pre=model.predict({'input':predict_ppl}, batch_size=128)
pre = pre['softmax3'].argmax(-1)
if is_ranking:
  pre = data.translate_ranking(pre)
  
data.save_predictions(pre, './ppl_result2.txt')


