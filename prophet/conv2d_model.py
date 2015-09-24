# vim: set fileencoding=utf-8 :
#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function

import os

from keras.optimizers import SGD, RMSprop, Adagrad
from keras.callbacks import ModelCheckpoint
from prophet.common import weibo_loss, weibo_loss_weighted, weibo_loss_scaled_weighted, WeiboPrecisionCallback, build_precisio_stack, detect_nan

from prophet.data import WeiboDataset
from prophet.models import *
import theano
import theano.compile
from theano.compile import monitormode

                    
dim=100
    
save_dir = "gen_nn_model"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
print("Training conv2d model")
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
dataset.load_words_vec(word_vec_filename, phrase_filenames, max_len=None)

print('Generating training/validation/predicting data')
train_gt = dataset.get_training_data_gt_matrix()
val_gt = dataset.get_validation_data_gt_matrix()
print("The max len of words is: ", dataset.get_missing_info(is_valid=False, is_predict=False, is_max_len=True, is_print=False))

train_words = dataset.get_words_vec_training_data_matrix(is_conv_2d=True)
#train_ppl = dataset.get_ppl_training_data_np()
val_words = dataset.get_words_vec_validation_data_matrix(is_conv_2d=True)
#val_ppl = dataset.get_ppl_validation_data_np()
predict_words = dataset.get_words_vec_predict_data_matrix(is_conv_2d=True)
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
  
print('Building the model')
vec_dim=100
model = build_conv2d_model(words=dataset.max_len())

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss=weibo_loss_scaled_weighted, optimizer=sgd, other_func_init=build_precisio_stack)#, theano_mode=theano.compile.MonitorMode(post_func=detect_nan).excluding(
    #'local_elemwise_fusion', 'inplace'))
#model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

print("Traing the model")
checkpoint = ModelCheckpoint(save_dir+"/combine_state.full_t.pkl", save_best_only=False)
precision = WeiboPrecisionCallback()
print(train_words.shape)
model.fit(train_words, train_gt, batch_size=256, nb_epoch=121, show_accuracy=True, callbacks=[checkpoint, precision], validation_data=( val_words, val_gt), shuffle=True)
train_res = model.predict(train_words, batch_size=128)
val_res = model.predict(val_words, batch_size=128)
import numpy
dataset._train_reader.save_data(numpy.append(train_res, val_res, 0), "./combine_val_result.txt")


print("predict shape: ", predict_words.shape)
pre=model.predict(predict_words, batch_size=128)
dataset.save_predictions(pre, './combine_result.txt')
exit(1)

