# vim: set fileencoding=utf-8 :
#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function

import os

from keras.optimizers import SGD, RMSprop, Adagrad
from keras.callbacks import ModelCheckpoint
from prophet.common import *

from prophet.data import WeiboDataset
from prophet.models import *
import theano
import theano.compile
from theano.compile import monitormode

from theano.sandbox.cuda import dnn

print("cnn gpu status: ", dnn.dnn_available())
#exit(1)
#model = build_conv2d_model(words=140, saved_filename="gen_nn_model/conv2d_state.full_t.pkl")
# w_limits = [1, 2, 3, 4, 5, 8, 10, 15, 20, 40, 100, 200, 300, 500, 1000]
# counting = [0 for n in w_limits]
# w_vals = model.layers[-1].W.get_value(borrow=True).flatten().tolist()
# for data in w_vals:
#   for idx, limit in enumerate(w_limits):
#     if data <= limit:
#       counting[idx] += 1
#       break
# for limit, n in zip(w_limits, counting):
#   print(" %d limit has %d numbers" %(limit, n) )
#dense_output_f = theano.function([model.get_input(train=False)], model.layers[0].get_output(train=False))

                    
dim=100
is_ranking = True
    
save_dir = "gen_nn_model"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
print("Training conv2d model")
if is_ranking:
  dataset = WeiboDataset(ranking_func=rank_limit)
else:
  dataset = WeiboDataset()
print('Loading the data...')
dataset.load_data(
             ("./data2/weibo_train_data.txt", "./gen_data2/weibo_train_data_text.txt.jian.words"),
             ("./data2/weibo_predict_data.txt", "./gen_data2/weibo_predict_data_text.txt.jian.words")
             )
print('Loading word vector model')
word_vec_filename="./gen_model2/vec_state_s100_p0_w5_t1.all"
#phrase_filenames="./gen_model2/vec_state_s100_p1_w5_t1.all.phrase0"
phrase_filenames=None
dataset.load_words_vec(word_vec_filename, phrase_filenames, max_len=140)

print('Generating training/validation/predicting data')
train_gt = dataset.get_training_data_gt_matrix(is_ranking=is_ranking)
val_gt = dataset.get_validation_data_gt_matrix(is_ranking=is_ranking)
print("The max len of words is: ", dataset.get_missing_info(is_valid=False, is_predict=False, is_max_len=True, is_print=False))

train_words = dataset.get_words_vec_training_data_matrix(is_conv_2d=True)
#train_words = dataset.get_words_vec_training_data_np(is_conv_2d=True)
#train_ppl = dataset.get_ppl_training_data_np()
val_words = dataset.get_words_vec_validation_data_matrix(is_conv_2d=True)
#val_words = dataset.get_words_vec_validation_data_np(is_conv_2d=True)
#val_ppl = dataset.get_ppl_validation_data_np()
predict_words = dataset.get_words_vec_predict_data_matrix(is_conv_2d=True)
#predict_words = dataset.get_words_vec_predict_data_np(is_conv_2d=True)
#predict_ppl = dataset.get_ppl_predict_data_np()
# import numpy as np
# if np.isnan(train_words).any():
#   print("train words contains nan")
#   exit(1)
# if np.isnan(val_words).any():
#   print("val_words contains nan")
#   exit(1)
# if np.isnan(predict_words).any():
#   print("predict_words contains nan")
#   exit(1)
#train_words = dataset.get_words_vec_training_data_np(is_conv_2d=True)
#print(train_words.shape)
#ret = dense_output_f(train_words)
#for line in ret.tolist():
#  print(line[0:10])
#exit(1)
#print(val_words[0:10,:,10,:])
  
print('Building the model')
vec_dim=100
model = build_conv2d_model(words=dataset.max_len(), is_ranking=is_ranking)
if is_ranking:
  lr_v = 0.001
else:
  lr_v = 0.05
sgd = SGD(lr=lr_v, decay=1e-6, momentum=0.9, nesterov=True)

if is_ranking:
  obj = RankedWeiboLoss(dataset)
  loss_func = obj.calculate_loss
else:
  loss_func = weibo_loss_weighted
  
model.compile(loss=loss_func, optimizer=sgd, other_func_init=build_precisio_stack)#, theano_mode=theano.compile.MonitorMode(post_func=detect_nan).excluding(
#model.compile(loss="mse", optimizer=sgd, other_func_init=build_precisio_stack)#, theano_mode=theano.compile.MonitorMode(post_func=detect_nan).excluding(
    #'local_elemwise_fusion', 'inplace'))
#model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

print("Traing the model")
checkpoint = ModelCheckpoint(save_dir+"/conv2d_state.full_t.pkl", save_best_only=False)
#precision = WeiboPrecisionCallback(val_saved_filename="./val_predictions.txt")
if is_ranking:
  precision = WeiboPrecisionCallback(1, 1, dataset_ranking=dataset, val_saved_filename="./val_predictions.txt")
else:
  precision = WeiboPrecisionCallback(1, 1, val_saved_filename="./val_predictions.txt")
  
print(train_words.shape)
model.fit(train_words, train_gt, batch_size=256, nb_epoch=121, show_accuracy=True, callbacks=[checkpoint, precision], validation_data=( val_words, val_gt), shuffle=True)
train_res = model.predict(train_words, batch_size=128)
val_res = model.predict(val_words, batch_size=128)
if is_ranking:
  train_res = dataset.translate_ranking(train_res)
  val_res = dataset.translate_ranking(val_res)
import numpy
dataset._train_reader.save_data(numpy.append(train_res, val_res, 0), "./conv2d_val_result.txt")


print("predict shape: ", predict_words.shape)
pre=model.predict(predict_words, batch_size=128)
if is_ranking:
  pre = dataset.translate_ranking(pre)
dataset.save_predictions(pre, './conv2d_result.txt')
exit(1)

