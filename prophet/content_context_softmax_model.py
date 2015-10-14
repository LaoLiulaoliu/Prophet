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
from keras.models import Graph
from keras.layers.core import *
from prophet.common import *

from prophet.data import WeiboDataset
from prophet.models import *

dim=100
is_ranking=True
is_categorical=True
    
save_dir = "gen_nn_model"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
if is_ranking:
  dataset = WeiboDataset(10000, ranking_func=rank_limit)
else:
  dataset = WeiboDataset(100000)
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
train_gt = dataset.get_training_data_gt_matrix(is_ranking=is_ranking, is_categorical=is_categorical)
#train_gt = dataset.get_training_data_gt_np(is_ranking=is_ranking, is_categorical=is_categorical)
#print(train_gt.shape)
#print(train_gt[0].shape)
#exit(1)
#print(dataset.get_training_data_gt_np(0, is_ranking=is_ranking, is_categorical=is_categorical))
#print(dataset.get_training_data_gt_np(0, is_ranking=is_ranking, is_categorical=is_categorical).shape)
#print(train_gt[0])
#print(train_gt[0].shape)
val_gt = dataset.get_validation_data_gt_matrix(is_ranking=is_ranking, is_categorical=is_categorical)
#val_gt = dataset.get_validation_data_gt_np(is_ranking=is_ranking, is_categorical=is_categorical)
print("The max len of words is: ", dataset.get_missing_info(is_valid=False, is_predict=False, is_max_len=True, is_print=False))

train_words = dataset.get_words_vec_training_data_matrix()
#train_words = dataset.get_words_vec_training_data_np()
#print(dataset.get_words_vec_training_data_np(0))
#print(dataset.get_words_vec_training_data_np(0).shape)
#print(train_words[0])
#print(train_words[0].shape)
val_words = dataset.get_words_vec_validation_data_matrix()
#val_words = dataset.get_words_vec_validation_data_np()
predict_words = dataset.get_words_vec_predict_data_matrix()

print('Building the model')
#model = build_content_context_model_2_lstm(max_len = dataset.max_len())
max_norm_v = 29
hidden1=1024
in_dim=dim
max_len=dataset.max_len()
model = Graph()
#print('shape: ', train_words.shape[1:], len(train_words.shape[1:]), type(train_words.shape[1:]))
#model.add_input(name="input", input_shape = train_words.shape[1][1:], dtype='float')
model.add_input(name="input", input_shape = train_words.shape[1:], dtype='float')
#print(model.inputs['input'].input_shape, model.inputs['input'].output_shape)
#print(len(model.inputs['input'].input_shape))
#print(model.get_config(verbose=1))
#model.add_node(LSTM(hidden1, input_dim=in_dim, return_sequences=True, input_length=max_len), name="lstm1", input='input')
model.add_node(LSTM(1024, return_sequences=True ), name="lstm1", input='input')
model.add_node(Dropout(0.5), name='drop1', input='lstm1')
model.add_node(LSTM(1024, return_sequences=False), input='drop1', name='lstm2')
model.add_node(Dropout(0.5), name='drop2', input='lstm2')
model.add_node(Dense(4096, init='uniform', activation="relu",
                  W_regularizer=l2(0.01), W_constraint = maxnorm(max_norm_v)), name='dense1', input='drop2')
model.add_node(Dropout(0.6), name='drop3', input='dense1')
model.add_node(Dense(4096, init='uniform', activation="relu",
                  W_regularizer=l2(0.01), W_constraint = maxnorm(max_norm_v)), name='dense2', input='drop3')
model.add_node(Dropout(0.6), name='drop4', input='dense2')
max_ranks_len = dataset.max_ranks()
model.add_node(Dense(max_ranks_len, init="normal", activation='softmax',
                  W_regularizer=l2(0.01), W_constraint = maxnorm(max_norm_v)), name='softmax_f', input='drop4')
model.add_node(Dense(max_ranks_len, init="normal", activation='softmax',
                  W_regularizer=l2(0.01), W_constraint = maxnorm(max_norm_v)), name='softmax_c', input='drop4')
model.add_node(Dense(max_ranks_len, init="normal", activation='softmax',
                  W_regularizer=l2(0.01), W_constraint = maxnorm(max_norm_v)), name='softmax_l', input='drop4')
model.add_node(Reshape(dims=(3, max_ranks_len)), inputs=['softmax_f', 'softmax_c', 'softmax_l'], merge_mode='concat', concat_axis=1, name='softmax3_pre', create_output=False)
model.add_output(input='softmax3_pre', name='softmax3')



sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
if is_ranking:
  obj = RankedWeiboLoss(dataset)
  loss_func = obj.categorical_crossentropy
  model.compile(loss={'softmax3':loss_func}, optimizer=sgd, other_func_init=build_precisio_stack_softmax, weighted_method="multi")
else:
  loss_func = weibo_loss_total_scaled_weighted
  model.compile(loss=loss_func, optimizer=sgd, other_func_init=build_precisio_stack)

#model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

print("Traing the model")
checkpoint = ModelCheckpoint(save_dir+"/content_context_state.full_t.pkl", save_best_only=False)
if is_ranking:
  precision = WeiboPrecisionCallback(1, 1, dataset_ranking=dataset, val_saved_filename="./test_saved_ppl_softmax.val")
else:
  precision = WeiboPrecisionCallback(1, 1)

model.fit({'input':train_words, 'softmax3':train_gt}, batch_size=128, nb_epoch=121, callbacks=[checkpoint, precision], validation_data={'input':val_words, 'softmax3':val_gt})

print("predict shape: ", predict_words.shape)
pre=model.predict({'input':predict_words}, batch_size=128)
pre = pre['softmax3'].argmax(-1)

if is_ranking:
  pre = dataset.translate_ranking(pre)

dataset.save_predictions(pre, './content_context_softmax.txt')
exit(1)

