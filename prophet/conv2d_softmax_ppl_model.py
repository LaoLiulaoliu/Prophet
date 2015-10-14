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
  dataset = WeiboDataset(1000, ranking_func=rank_limit)
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
val_gt = dataset.get_validation_data_gt_matrix(is_ranking=is_ranking, is_categorical=is_categorical)
print("The max len of words is: ", dataset.get_missing_info(is_valid=False, is_predict=False, is_max_len=True, is_print=False))

train_ppl = dataset.get_ppl_training_data_matrix()
val_ppl = dataset.get_ppl_validation_data_matrix()
predict_ppl = dataset.get_ppl_predict_data_matrix()

train_words = dataset.get_words_vec_training_data_matrix(is_conv_2d=True)
val_words = dataset.get_words_vec_validation_data_matrix(is_conv_2d=True)
predict_words = dataset.get_words_vec_predict_data_matrix(is_conv_2d=True)

print('Building the model')
#model = build_content_context_model_2_lstm(max_len = dataset.max_len())
max_norm_v = 29
hidden1=1024
in_dim=dim
max_len=dataset.max_len()
model = Graph()
model.add_input(name="words_input", input_shape = train_words.shape[1:], dtype='float')
model.add_input(name="ppl_input", input_shape = train_ppl.shape[1:], dtype='int32')

nb_filter = 64
ngrams = [2,3,4,5,6,7, 8]
nb_row = max_len
nb_col = dim
w_decay = 0.0005

conv_output_names = []  
for nb_row in ngrams:
  # convolution
  conv_name = "conv2d_ng_" + str(nb_row)
  model.add_node(Convolution2D(nb_filter, nb_row, nb_col, init="normal", subsample=(1,nb_col),
                    W_regularizer=l2(w_decay), W_constraint = maxnorm(29), input_shape=(1, nb_row, nb_col)),
            name=conv_name, input='words_input')
  act_name = "conv2d_ng_ac_" + str(nb_row)
  model.add_node(Activation('relu'), name=act_name, input=conv_name)
  pool_size = 4
  stride_size = 1  
  pool_name = "conv2d_ng_pool_" + str(nb_row)
  model.add_node(MaxPooling2D(pool_size=(pool_size, 1), stride=(stride_size, 1)),
            name=pool_name, input=act_name)
  flatten_name = "conv2d_ng_flatten_" + str(nb_row)
  conv_output_names.append(flatten_name)
  model.add_node(Flatten(), name=flatten_name, input=pool_name)

model.add_node(Dropout(0.7), name='conv_drop1', inputs=conv_output_names, merge_mode='concat', concat_axis=1)
model.add_node(Dense(4096, init="uniform", activation="relu",
                     W_regularizer=l2(0.01), W_constraint = maxnorm(max_norm_v)),
               name="conv2_dense1", input='conv_drop1'
               )
model.add_node(Dropout(0.7), name='conv_drop2', input='conv2_dense1')
model.add_node(Dense(4096, init="uniform", activation="relu",
                     W_regularizer=l2(0.01), W_constraint = maxnorm(max_norm_v)),
               name="conv2_dense2", input='conv_drop2'
               )

# embeding the ppls
max_ppl = dataset.get_ppl_max_count()
dim_proj = 300 
model.add_node(Embedding(max_ppl, dim_proj, init="uniform", input_length=1), 
               name='ppl_embeding', input='ppl_input')
model.add_node(Flatten(), name='ppl_flatten', input='ppl_embeding')

model.add_node(Dropout(0.7), name='mix_drop1', inputs=['ppl_flatten','conv2_dense2'], merge_mode='concat', concat_axis=1)
model.add_node(Dense(4096, init="uniform", activation="relu",
                     W_regularizer=l2(0.01), W_constraint = maxnorm(max_norm_v)),
               name="mix_dense1", input='mix_drop1'
               )
model.add_node(Dropout(0.8), name='mix_drop2', input='mix_dense1')

# now, let's do softmax
max_ranks_len = dataset.max_ranks()
model.add_node(Dense(max_ranks_len, init="normal", activation='softmax',
                  W_regularizer=l2(0.01), W_constraint = maxnorm(max_norm_v)), name='softmax_f', input='mix_drop2')
model.add_node(Dense(max_ranks_len, init="normal", activation='softmax',
                  W_regularizer=l2(0.01), W_constraint = maxnorm(max_norm_v)), name='softmax_c', input='mix_drop2')
model.add_node(Dense(max_ranks_len, init="normal", activation='softmax',
                  W_regularizer=l2(0.01), W_constraint = maxnorm(max_norm_v)), name='softmax_l', input='mix_drop2')
model.add_node(Reshape(dims=(3, max_ranks_len)), inputs=['softmax_f', 'softmax_c', 'softmax_l'], merge_mode='concat', concat_axis=1, name='softmax3_pre', create_output=False)
model.add_output(input='softmax3_pre', name='softmax3')



sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss={'softmax3':'categorical_crossentropy'}, optimizer=sgd, other_func_init=build_precisio_stack_softmax)
#exit(1)
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

model.fit({'words_input':train_words, 'ppl_input':train_ppl, 'softmax3':train_gt}, 
          batch_size=128, nb_epoch=121, callbacks=[checkpoint, precision], 
          validation_data={'words_input':val_words, 'ppl_input':val_ppl, 'softmax3':val_gt})

print("predict shape: ", predict_words.shape)
pre=model.predict({'words_input':predict_words, 'ppl_input':predict_ppl}, batch_size=128)
pre = pre['softmax3'].argmax(-1)

if is_ranking:
  pre = dataset.translate_ranking(pre)

dataset.save_predictions(pre, './conv2d_mix_softmax_softmax.txt')
exit(1)

