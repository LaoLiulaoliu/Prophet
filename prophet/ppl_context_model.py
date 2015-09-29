# vim: set fileencoding=utf-8 :
#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function

import os
from gensim.test.test_parsing import dataset
prophet_root = '/home/xijing/Work/weibo/Prophet'
import sys
sys.path.insert(0, prophet_root)


from keras.optimizers import SGD, RMSprop, Adagrad
from keras.callbacks import ModelCheckpoint

from prophet.metric import WeiboPrecision
from prophet.common import *

from prophet.data import WeiboDataset
from prophet.models import *

max_features = 1000000  # vocabulary size: top 50,000 most common words in data
skip_top = 0  # ignore top 100 most common words
nb_epoch = 1
dim_proj = 300  # embedding space dimension
dim_output = 3 # the output predicting values.

save = True
is_ranking = True
#load_model = False
#train_model = True

save_dir = "gen_nn_model"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
model_load_fname = "ppl_context.pkl"
model_save_fname = "ppl_context.pkl"

print('Loading the data...')
import time
start = time.time()
if is_ranking:
  data = WeiboDataset(ranking_func=rank_limit)
else:
  data = WeiboDataset()
data.load_data("./data2/weibo_train_data.txt", "./data2/weibo_predict_data.txt", is_init_ppl_standard=False)


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
model = build_ppl_context_model(max_features, dim_proj, dim_output)
#model.compile(loss=weibo_loss, optimizer='rmsprop')
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
#sgd = SGD(lr=2, decay=1e-6, momentum=0.9, nesterov=True)
#use weibo_loss_weighted with learning rate 2 is good.
#model.compile(loss=weibo_loss_weighted, optimizer=sgd)
obj = RankedWeiboLoss(data)
model.compile(loss=obj.calculate_loss, optimizer=sgd, other_func_init=build_precisio_stack)
#model.compile(loss="mse", optimizer=sgd)

checkpoint = ModelCheckpoint(save_dir+"/ppl_context_state2.full_t.10.f10.pkl", save_best_only=False)
if is_ranking:
  precision = WeiboPrecisionCallback(1, 1, dataset_ranking=data)
else:
  precision = WeiboPrecisionCallback(1, 1)

model.fit(train_ppl, train_gt, batch_size=256, nb_epoch=120, show_accuracy=True, callbacks=[checkpoint, precision], validation_data=(val_ppl, val_gt))
#model.fit(train_ppl, train_gt, batch_size=256, nb_epoch=120, show_accuracy=True, callbacks=[checkpoint], validation_data=(val_ppl, val_gt))

pre=model.predict(val_ppl, batch_size=128)
print("validation weibo acc: ", WeiboPrecision.precision_match(val_gt, pre))

pre=model.predict(predict_ppl, batch_size=128)
data.save_predictions(pre, './ppl_result2.txt')


