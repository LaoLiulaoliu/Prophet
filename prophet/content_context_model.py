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
from prophet.common import weibo_loss, weibo_loss_weighted, weibo_loss_scaled_weighted, WeiboPrecisionCallback, build_precisio_stack

from prophet.data import WeiboDataset
from prophet.models import *

dim=100
    
save_dir = "gen_nn_model"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    

dataset = WeiboDataset()
print('Loading the data...')
dataset.load_data(
             ("./data2/weibo_train_data.txt", "./gen_data2/weibo_train_data_text.txt.jian.words"),
             ("./data2/weibo_predict_data.txt", "./gen_data2/weibo_predict_data_text.txt.jian.words")
             )
print('Loading word vector model')
word_vec_filename="./gen_model2/vec_state_s100_p1_w5_t1.all"
phrase_filenames="./gen_model2/vec_state_s100_p1_w5_t1.all.phrase0"
dataset.load_words_vec(word_vec_filename, phrase_filenames, max_len=None)

print('Generating training/validation/predicting data')
train_gt = dataset.get_training_data_gt_matrix()
val_gt = dataset.get_validation_data_gt_matrix()
print("The max len of words is: ", dataset.get_missing_info(is_valid=False, is_predict=False, is_max_len=True, is_print=False))

train_words = dataset.get_words_vec_training_data_matrix()
val_words = dataset.get_words_vec_validation_data_matrix()
predict_words = dataset.get_words_vec_predict_data_matrix()

print('Building the model')
model = build_content_context_model_2_lstm()
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss=weibo_loss_scaled_weighted, optimizer=sgd, other_func_init=build_precisio_stack)
#model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

print("Traing the model")
checkpoint = ModelCheckpoint(save_dir+"/content_context_state.full_t.pkl", save_best_only=False)
precision = WeiboPrecisionCallback()
model.fit(train_words, train_gt, batch_size=256, nb_epoch=121, show_accuracy=True, callbacks=[checkpoint, precision], validation_data=(val_words, val_gt))

print("predict shape: ", predict_words.shape)
pre=model.predict(predict_words, batch_size=128)
dataset.save_predictions(pre, './ppl_result.txt')
exit(1)

