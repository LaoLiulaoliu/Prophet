# vim: set fileencoding=utf-8 :
#!/usr/bin/env python


import prophet

reader = prophet.WeiboReader()
reader.load_data("./data/weibo_train_data.txt")
reader.print_dataset_info()
reader.print_top_list()

#reader.load_data("./data/weibo_predict_data.txt")
#reader.print_dataset_info()