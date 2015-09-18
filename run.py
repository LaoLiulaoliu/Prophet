# vim: set fileencoding=utf-8 :
#!/usr/bin/env python


import prophet

#reader = prophet.WeiboReader()
#reader.load_data("./data/weibo_train_data.txt")
#reader.print_dataset_info()
#reader.print_top_list()

reader = prophet.WeiboReader()
reader.load_data("./data/weibo_predict_data.txt")
data_list = reader.data()
predictions = []
for data in data_list:
  predictions.append((0,0,0))
reader.save_data(predictions, "./result.txt")
#reader.load_data("./data/weibo_predict_data.txt")
#reader.print_dataset_info()