# vim: set fileencoding=utf-8 :
#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function

import os

from keras.optimizers import SGD, RMSprop, Adagrad
from keras.callbacks import ModelCheckpoint
from prophet.common import weibo_loss, weibo_loss_weighted, weibo_loss_scaled_weighted, WeiboPrecisionCallback, build_precisio_stack

from prophet.data import WeiboDataset
from prophet.models import *
import theano
import theano.compile
from theano.compile import monitormode
import numpy

def detect_nan(i, node, fn):
    for output in fn.outputs:
        if (not isinstance(output[0], numpy.random.RandomState) and
            numpy.isnan(output[0]).any()):
            print('*** NaN detected *** i: ', i)
            theano.printing.debugprint(node)
            print('Inputs : %s' % [input[0] for input in fn.inputs])
            #print('numpy result is %s' % ([numpy.asarray(input[0]).tolist() for input in fn.inputs]))
            print('Outputs: %s' % [output[0] for output in fn.outputs])
            
            for vec in numpy.asarray(fn.outputs[0][0]).tolist():  
              if numpy.isnan(numpy.array(vec)):
                print(vec)
              
            #print('numpy result is %s' % ([numpy.asarray(output[0]).tolist() for output in fn.outputs]))
            exit(1)
                    
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
dataset.load_words_vec(word_vec_filename, phrase_filenames, max_len=10)

print('Generating training/validation/predicting data')
train_gt = dataset.get_training_data_gt_np()
val_gt = dataset.get_validation_data_gt_np()
print("The max len of words is: ", dataset.get_missing_info(is_valid=False, is_predict=False, is_max_len=True, is_print=False))

train_words = dataset.get_words_vec_training_data_np()
train_ppl = dataset.get_ppl_training_data_np()
val_words = dataset.get_words_vec_validation_data_np()
val_ppl = dataset.get_ppl_validation_data_np()
predict_words = dataset.get_words_vec_predict_data_np()
predict_ppl = dataset.get_ppl_predict_data_np()

print('Building the model')
vec_dim=100
model = build_combine_model(dataset.get_ppl_max_count(), vec_dim=vec_dim)

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss=weibo_loss_scaled_weighted, optimizer=sgd, other_func_init=build_precisio_stack) #, theano_mode=theano.compile.MonitorMode(post_func=detect_nan))
#model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

print("Traing the model")
checkpoint = ModelCheckpoint(save_dir+"/combine_state.full_t.pkl", save_best_only=False)
precision = WeiboPrecisionCallback()
model.fit([train_ppl, train_words], train_gt, batch_size=256, nb_epoch=121, show_accuracy=True, callbacks=[checkpoint, precision], validation_data=([val_ppl, val_words], val_gt))
train_res = model.predict([train_ppl, train_words], batch_size=128)
val_res = model.predict([val_ppl, val_words], batch_size=128)
import numpy
dataset._train_reader.save_data(numpy.append(train_res, val_res, 0), "./combine_val_result.txt")


print("predict shape: ", predict_words.shape)
pre=model.predict([predict_ppl, predict_words], batch_size=128)
dataset.save_predictions(pre, './combine_result.txt')
exit(1)

