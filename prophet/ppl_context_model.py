# vim: set fileencoding=utf-8 :
#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function

import os

from keras.optimizers import SGD, RMSprop, Adagrad
from keras.callbacks import ModelCheckpoint

from prophet.metric import WeiboPrecision
from prophet.common import weibo_loss, weibo_loss_weighted, weibo_loss_scaled_weighted, build_percision_funcs, WeiboPrecisionCallback, build_precisio_stack

from prophet.data import WeiboDataset
from prophet.models import *

max_features = 1000000  # vocabulary size: top 50,000 most common words in data
skip_top = 0  # ignore top 100 most common words
nb_epoch = 1
dim_proj = 300  # embedding space dimension
dim_output = 3 # the output predicting values.

save = True
#load_model = False
#train_model = True

save_dir = "gen_nn_model"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
model_load_fname = "ppl_context.pkl"
model_save_fname = "ppl_context.pkl"

print('Loading the data...')
data = WeiboDataset()
data.load_data("./data/weibo_train_data.txt", "./data/weibo_predict_data.txt")


train_gt = data.get_training_data_gt_np()
train_ppl = data.get_ppl_training_data_np()
val_gt = data.get_validation_data_gt_np()
val_ppl = data.get_ppl_validation_data_np()

predict_ppl = data.get_ppl_predict_data_np()

missing_info = data.get_missing_info(is_max_len=False)

print("-- validation missing %d users, %d weibo" % (missing_info['val'], missing_info['val_c']))
print("-- predict data missing %d users, %d weibo" % (missing_info['pre'], missing_info['pre_c']))
  
max_features = data.get_ppl_max_count()

print('Build model...')
model = build_ppl_context_model(max_features, dim_proj, dim_output)
#model.compile(loss=weibo_loss, optimizer='rmsprop')
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
#sgd = SGD(lr=2, decay=1e-6, momentum=0.9, nesterov=True)
#use weibo_loss_weighted with learning rate 2 is good.
#model.compile(loss=weibo_loss_weighted, optimizer=sgd)
model.compile(loss=weibo_loss_scaled_weighted, optimizer=sgd, other_func_init=build_precisio_stack)

checkpoint = ModelCheckpoint(save_dir+"/ppl_context_state.full_t.10.f10.pkl", save_best_only=False)
precision = WeiboPrecisionCallback()
model.fit(train_ppl, train_gt, batch_size=256, nb_epoch=120, show_accuracy=True, callbacks=[checkpoint, precision], validation_data=(val_ppl, val_gt))

pre=model.predict(val_ppl, batch_size=128)
print("validation weibo acc: ", WeiboPrecision.precision_match(val_gt, pre))

pre=model.predict(predict_ppl, batch_size=128)
data.save_predictions(pre, './ppl_result.txt')


