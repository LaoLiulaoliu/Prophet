# vim: set fileencoding=utf-8 :
#!/usr/bin/env python


import prophet
training_filename="./data/weibo_train_data.txt"
predict_filename="./data/weibo_predict_data.txt"

reader = prophet.WeiboReader()
reader.load_data(training_filename)
print("========================================================")
reader.print_dataset_info()
print("========================================================")
reader.print_top_list()
print("========================================================")
reader.print_rank_info()
print("========================================================")
from prophet.ppl_idx_table import PplIdxTable
train_data = reader.get_training_data()
idx_table = PplIdxTable()
idx_table.create_ppls_table(train_data, lambda info: info[0])
val_data = reader.get_validation_data()
idx_table.reset_missing()
idx_table.get_ppls_idx(val_data, lambda info: info[0])
print("----validation missing: %d users" % (idx_table.get_missing_uniq_ppl()) )
idx_table.reset_missing()
rd2 = prophet.WeiboReader()
rd2.load_data(predict_filename)
predict_data = rd2._data
idx_table.get_ppls_idx(predict_data, lambda info: info[0])
print("-- predict data missing %d users" % idx_table.get_missing_uniq_ppl())

idx_table.create_ppls_table(val_data, lambda info: info[0])
idx_table.reset_missing()
idx_table.get_ppls_idx(predict_data, lambda info: info[0])
print("-- with all training data, predict data missing %d users" % idx_table.get_missing_uniq_ppl())

#reader = prophet.WeiboReader()
#reader.load_data("./data/weibo_predict_data.txt")
#data_list = reader.data()
#predictions = []
#for data in data_list:
#  predictions.append((0,0,0))
#reader.save_data(predictions, "./result.txt")
#reader.load_data("./data/weibo_predict_data.txt")
#reader.print_dataset_info()