# vim: set fileencoding=utf-8 :
#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import theano
from six.moves import cPickle
import os, re, json

from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Flatten
from keras.models import Sequential 
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils, generic_utils
from keras.regularizers import l2
from prophet.weibo_reader import WeiboReader
from prophet.metric import WeiboPrecision
from prophet.common import weibo_loss, weibo_loss_weighted, weibo_loss_scaled_weighted
from keras.callbacks import ModelCheckpoint

max_features = 1000000  # vocabulary size: top 50,000 most common words in data
skip_top = 0  # ignore top 100 most common words
nb_epoch = 1
dim_proj = 10  # embedding space dimension
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
reader = WeiboReader()
reader.load_data("./data/weibo_train_data.txt")
train_data = reader.get_training_data()
#train_data = reader._data
train_gt = np.array([[info[3], info[4], info[5]] for info in train_data], dtype='float32')
# setup ppl to id indexing.
ppl_idx_table = {}
print("first uid: %s" % train_data[0][0])
for info in train_data:
  ppl_id = info[0]
  if ppl_id not in ppl_idx_table:
    ppl_idx_table[ppl_id] = len(ppl_idx_table)
    

train_ppl = np.array([[ppl_idx_table[info[0]]] for info in train_data ], dtype='int' )

val_data = reader.get_validation_data()
val_gt = np.array([[info[3], info[4], info[5]] for info in val_data], dtype='float32')
# default empty user using: 07fc721342df1a4c1992560b582992f8 vector
val_ppl = []
missing = {}
for info in val_data:
  ppl_id = info[0]
  if ppl_id in ppl_idx_table:
    val_ppl.append([ppl_idx_table[ppl_id]])
  else:
    #print("-- user id: %s is missing in index table" % ppl_id)
    missing[ppl_id] = 1
    val_ppl.append([0])
print("-- missing %d users" % len(missing))
val_ppl = np.array(val_ppl, dtype='float32')
    
rd2 = WeiboReader()
rd2.load_data("./data/weibo_predict_data.txt")
predict_data = rd2._data
predict_ppl = []
missing = {}
for info in predict_data:
  ppl_id = info[0]
  if ppl_id not in ppl_idx_table:
    missing[ppl_id] = 1
    predict_ppl.append([0])
  else:
    predict_ppl.append([ppl_idx_table[ppl_id]])
predict_ppl = np.array(predict_ppl, dtype='float32')

print("-- predict data missing %d users" % len(missing))
  
max_features = len(ppl_idx_table)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, dim_proj, init="uniform"))
model.add(Flatten())
model.add(Dense(dim_proj, dim_output, init="uniform", activation="linear",
                W_regularizer=l2(0.01)))
#model.compile(loss=weibo_loss, optimizer='rmsprop')
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
#use weibo_loss_weighted with learning rate 2 is good.
#model.compile(loss=weibo_loss_weighted, optimizer=sgd)
model.compile(loss=weibo_loss_scaled_weighted, optimizer=sgd)

checkpoint = ModelCheckpoint(save_dir+"/ppl_context_state.full_t.10.f10.pkl", save_best_only=False)
model.fit(train_ppl, train_gt, batch_size=256, nb_epoch=10, show_accuracy=True, callbacks=[checkpoint])
train_pre=model.predict(train_ppl, batch_size=128)
print("traing weibo acc: ", WeiboPrecision.precision_match(train_gt, train_pre))

print("validation acc: ", model.evaluate(val_ppl, val_gt, 128, show_accuracy=True))
pre=model.predict(val_ppl, batch_size=128)
print("validation weibo acc: ", WeiboPrecision.precision_match(val_gt, pre))

pre=model.predict(predict_ppl, batch_size=128)
rd2.save_data(pre, './ppl_result.txt')
exit(1)
#print(train_ppl)
checkpoint = ModelCheckpoint(save_dir+"/ppl_context_state1.pkl", save_best_only=False)
model.fit(train_ppl, train_gt, batch_size=128, nb_epoch=1, show_accuracy=True, callbacks=[checkpoint])
print("validation acc: ", model.evaluate(val_ppl, val_gt, 128, show_accuracy=True))
pre=model.predict(val_ppl, batch_size=128)
print("validation weibo acc: ", WeiboPrecision.precision_match(val_gt, pre))
checkpoint = ModelCheckpoint(save_dir+"/ppl_context_state2.pkl", save_best_only=False)
model.fit(train_ppl, train_gt, batch_size=128, nb_epoch=1, show_accuracy=True, callbacks=[checkpoint])
print("validation acc: ", model.evaluate(val_ppl, val_gt, 128, show_accuracy=True))
pre=model.predict(val_ppl, batch_size=128)
print("validation weibo acc: ", WeiboPrecision.precision_match(val_gt, pre))
checkpoint = ModelCheckpoint(save_dir+"/ppl_context_state3.pkl", save_best_only=False)
model.fit(train_ppl, train_gt, batch_size=128, nb_epoch=1, show_accuracy=True, callbacks=[checkpoint])
print("validation acc: ", model.evaluate(val_ppl, val_gt, 128, show_accuracy=True))
pre=model.predict(val_ppl, batch_size=128)
print("validation weibo acc: ", WeiboPrecision.precision_match(val_gt, pre))
checkpoint = ModelCheckpoint(save_dir+"/ppl_context_state4.pkl", save_best_only=False)
model.fit(train_ppl, train_gt, batch_size=128, nb_epoch=1, show_accuracy=True, callbacks=[checkpoint])
print("validation acc: ", model.evaluate(val_ppl, val_gt, 128, show_accuracy=True))
pre=model.predict(val_ppl, batch_size=128)
print("validation weibo acc: ", WeiboPrecision.precision_match(val_gt, pre))
checkpoint = ModelCheckpoint(save_dir+"/ppl_context_state5.pkl", save_best_only=False)
model.fit(train_ppl, train_gt, batch_size=128, nb_epoch=1, show_accuracy=True, callbacks=[checkpoint])
print("validation acc: ", model.evaluate(val_ppl, val_gt, 128, show_accuracy=True))
pre=model.predict(val_ppl, batch_size=128)
print("validation weibo acc: ", WeiboPrecision.precision_match(val_gt, pre))
checkpoint = ModelCheckpoint(save_dir+"/ppl_context_state6.pkl", save_best_only=False)
model.fit(train_ppl, train_gt, batch_size=128, nb_epoch=1, show_accuracy=True, callbacks=[checkpoint])
print("validation acc: ", model.evaluate(val_ppl, val_gt, 128, show_accuracy=True))
checkpoint = ModelCheckpoint(save_dir+"/ppl_context_state7.pkl", save_best_only=False)
pre=model.predict(val_ppl, batch_size=128)
print("validation weibo acc: ", WeiboPrecision.precision_match(val_gt, pre))
model.fit(train_ppl, train_gt, batch_size=128, nb_epoch=1, show_accuracy=True, callbacks=[checkpoint])
print("validation acc: ", model.evaluate(val_ppl, val_gt, 128, show_accuracy=True))
pre=model.predict(val_ppl, batch_size=128)
print("validation weibo acc: ", WeiboPrecision.precision_match(val_gt, pre))
checkpoint = ModelCheckpoint(save_dir+"/ppl_context_state8.pkl", save_best_only=False)
model.fit(train_ppl, train_gt, batch_size=128, nb_epoch=1, show_accuracy=True, callbacks=[checkpoint])
print("validation acc: ", model.evaluate(val_ppl, val_gt, 128, show_accuracy=True))
pre=model.predict(val_ppl, batch_size=128)
print("validation weibo acc: ", WeiboPrecision.precision_match(val_gt, pre))
checkpoint = ModelCheckpoint(save_dir+"/ppl_context_state9.pkl", save_best_only=False)
model.fit(train_ppl, train_gt, batch_size=128, nb_epoch=1, show_accuracy=True, callbacks=[checkpoint])
print("validation acc: ", model.evaluate(val_ppl, val_gt, 128, show_accuracy=True))
pre=model.predict(val_ppl, batch_size=128)
print("validation weibo acc: ", WeiboPrecision.precision_match(val_gt, pre))
checkpoint = ModelCheckpoint(save_dir+"/ppl_context_state10.pkl", save_best_only=False)
model.fit(train_ppl, train_gt, batch_size=128, nb_epoch=1, show_accuracy=True, callbacks=[checkpoint])
print("validation acc: ", model.evaluate(val_ppl, val_gt, 128, show_accuracy=True))
pre=model.predict(val_ppl, batch_size=128)
print("validation weibo acc: ", WeiboPrecision.precision_match(val_gt, pre))

exit(1)
model.fit(train_ppl, train_gt, batch_size=128, nb_epoch=1, show_accuracy=True, callbacks=[checkpoint])
print(model.evaluate(val_ppl, val_gt, 128, show_accuracy=True))
pre=model.predict(val_ppl, batch_size=128)
print(WeiboPrecision.precision_match(pre, val_gt))
model.fit(train_ppl, train_gt, batch_size=128, nb_epoch=9, show_accuracy=True)
print(model.evaluate(val_ppl, val_gt, 128, show_accuracy=True))
pre=model.predict(val_ppl, batch_size=128)
print(WeiboPrecision.precision_match(pre, val_gt))
model.fit(train_ppl, train_gt, batch_size=128, nb_epoch=9, show_accuracy=True)
print(model.evaluate(val_ppl, val_gt, 128, show_accuracy=True))
pre=model.predict(val_ppl, batch_size=128)
print(WeiboPrecision.precision_match(pre, val_gt))
model.fit(train_ppl, train_gt, batch_size=128, nb_epoch=9, show_accuracy=True)
pre=model.predict(val_ppl, batch_size=128)
print(WeiboPrecision.precision_match(pre, val_gt))


