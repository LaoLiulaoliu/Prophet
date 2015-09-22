# vim: set fileencoding=utf-8 :
#!/usr/bin/env python

"""
  Do read/write the weibo content from/to train/prediction data.
  Created by: BlackCat@exadeep
"""

import os
from sets import Set
import operator
import datetime
import string

def comp_forward_count(item1, item2):
  int1 = int(item1[3])
  int2 = int(item2[3])
  if int1 > int2:
    return 1
  elif int1 == int2:
    return 0
  else:
    return -1
  
def comp_comment_count(item1, item2):
  int1 = int(item1[4])
  int2 = int(item2[4])
  if int1 > int2:
    return 1
  elif int1 == int2:
    return 0
  else:
    return -1

def comp_like_count(item1, item2):
  int1 = int(item1[5])
  int2 = int(item2[5])
  if int1 > int2:
    return 1
  elif int1 == int2:
    return 0
  else:
    return -1
  
  
def get_forward_count(item):
  return int(item[3])

def get_comment_count(item):
  return int(item[4])

def get_like_count(item):
  return int(item[5])

def get_date_info(item):
  t_list = item[2].split(" ")
  if len(t_list) > 1:
    # date and time.
    s = t_list[0].split("-")
    return datetime.date(int(s[0]), int(s[1]), int(s[2]))
  else:
    # date only
    s = item[2].split("-")
    return datetime.date(int(s[0]), int(s[1]), int(s[2]))

banlist=Set()
#banlist.add(".")
banlist.add(",")
#banlist.add("*")
banlist.add("-")
banlist.add("!")
banlist.add("`")
banlist.add("(")
banlist.add(u"●".encode('utf-8'))
banlist.add("'")
banlist.add("◡")
banlist.add("'")
banlist.add(")")
banlist.add(u"、".encode('utf-8'))
banlist.add("f")
banlist.add(u"。".encode('utf-8'))
banlist.add("+")
banlist.add("[")
banlist.add("]")
banlist.add("_")
banlist.add("h")
banlist.add("5")
banlist.add("~")
banlist.add("\\")
banlist.add("u")
banlist.add("6211")
#banlist.add("521")
banlist.add("53")
banlist.add("d")
banlist.add("73")
banlist.add("86")
banlist.add("7279")
banlist.add("522")
banlist.add("6709")
banlist.add("7684")
banlist.add("69")
banlist.add("59")
banlist.add("ee")
banlist.add("3010")
banlist.add("3002")
banlist.add(u"，".encode('utf-8'))
banlist.add(u"！".encode('utf-8'))
banlist.add(u"（".encode('utf-8'))
banlist.add("ch")
banlist.add(u"ù".encode('utf-8'))
banlist.add(u"）".encode('utf-8'))
banlist.add(u"¾".encode('utf-8'))
banlist.add(u"©".encode('utf-8'))
banlist.add(u"¶".encode('utf-8'))
banlist.add(u"«".encode('utf-8'))
banlist.add(u"Õ".encode('utf-8'))
banlist.add(u"½".encode('utf-8'))
banlist.add(u"Â".encode('utf-8'))
banlist.add(u"Ô".encode('utf-8'))
banlist.add(u"Í".encode('utf-8'))
banlist.add(u"¶".encode('utf-8'))
banlist.add(u"×".encode('utf-8'))
banlist.add(u"Ê".encode('utf-8'))
banlist.add(u"½".encode('utf-8'))
banlist.add(u"ð".encode('utf-8'))
banlist.add(u"µ".encode('utf-8'))
banlist.add(u"û".encode('utf-8'))
banlist.add(u"£".encode('utf-8'))
banlist.add(u"¨".encode('utf-8'))
banlist.add(u"æ".encode('utf-8'))
banlist.add(u"±".encode('utf-8'))
banlist.add(u"".encode('utf-8'))
banlist.add(u"æ".encode('utf-8'))
banlist.add(u"".encode('utf-8'))
banlist.add(u"".encode('utf-8'))
banlist.add(u"å".encode('utf-8'))
banlist.add(u"".encode('utf-8'))
banlist.add(u"¡".encode('utf-8'))
banlist.add(u"ï".encode('utf-8'))
banlist.add(u"¼".encode('utf-8'))
banlist.add(u"".encode('utf-8'))
banlist.add(u"å".encode('utf-8'))
banlist.add(u"".encode('utf-8'))
banlist.add(u"".encode('utf-8'))
banlist.add(u"å".encode('utf-8'))
banlist.add(u"¤".encode('utf-8'))
banlist.add(u"©".encode('utf-8'))
banlist.add(u"ä".encode('utf-8'))
banlist.add(u"¸".encode('utf-8'))
banlist.add(u"".encode('utf-8'))
banlist.add(u"â".encode('utf-8'))
banlist.add(u"".encode('utf-8'))
banlist.add(u"".encode('utf-8'))
banlist.add(u"â".encode('utf-8'))
banlist.add(u"".encode('utf-8'))
banlist.add(u"".encode('utf-8'))
banlist.add(u"è".encode('utf-8'))
banlist.add(u"¥".encode('utf-8'))
banlist.add(u"".encode('utf-8'))
banlist.add(u"é".encode('utf-8'))
banlist.add(u"".encode('utf-8'))
banlist.add(u"³".encode('utf-8'))
banlist.add(u"æ".encode('utf-8'))
banlist.add(u"±".encode('utf-8'))
banlist.add(u"".encode('utf-8'))
banlist.add(u"å".encode('utf-8'))
banlist.add(u"".encode('utf-8'))
banlist.add(u"".encode('utf-8'))
banlist.add(u"ä".encode('utf-8'))
banlist.add(u"¿".encode('utf-8'))
banlist.add(u"¡".encode('utf-8'))
banlist.add(u"æ".encode('utf-8'))
banlist.add(u"".encode('utf-8'))
banlist.add(u"¯".encode('utf-8'))
banlist.add(u"æ".encode('utf-8'))
banlist.add(u"".encode('utf-8'))
banlist.add(u"".encode('utf-8'))
banlist.add(u"æ".encode('utf-8'))
banlist.add(u"".encode('utf-8'))
banlist.add(u"¯".encode('utf-8'))
banlist.add(u"æ".encode('utf-8'))
banlist.add(u"".encode('utf-8'))
banlist.add(u"".encode('utf-8'))
banlist.add(u"é".encode('utf-8'))
banlist.add(u"".encode('utf-8'))
banlist.add(u"".encode('utf-8'))
banlist.add(u"å".encode('utf-8'))
banlist.add(u"".encode('utf-8'))
banlist.add(u"¬".encode('utf-8'))
banlist.add(u"å".encode('utf-8'))
banlist.add(u"".encode('utf-8'))
banlist.add(u"¸".encode('utf-8'))
banlist.add(u"ï".encode('utf-8'))
banlist.add(u"¼".encode('utf-8'))
banlist.add(u"".encode('utf-8'))
banlist.add(u"è".encode('utf-8'))
banlist.add(u"¥".encode('utf-8'))
banlist.add(u"".encode('utf-8'))
banlist.add(u"é".encode('utf-8'))
banlist.add(u"".encode('utf-8'))
banlist.add(u"³".encode('utf-8'))
banlist.add(u"é".encode('utf-8'))
banlist.add(u"".encode('utf-8'))
banlist.add(u"".encode('utf-8'))
banlist.add(u"è".encode('utf-8'))
banlist.add(u"".encode('utf-8'))
banlist.add(u"¶".encode('utf-8'))
banlist.add(u"è".encode('utf-8'))
banlist.add(u"½".encode('utf-8'))
banlist.add(u"¯".encode('utf-8'))
banlist.add(u"ä".encode('utf-8'))
banlist.add(u"»".encode('utf-8'))
banlist.add(u"¶".encode('utf-8'))
banlist.add(u"å".encode('utf-8'))
banlist.add(u"".encode('utf-8'))
banlist.add(u"¯".encode('utf-8'))
banlist.add(u"ä".encode('utf-8'))
banlist.add(u"¸".encode('utf-8'))
banlist.add(u"".encode('utf-8'))
banlist.add(u"æ".encode('utf-8'))
banlist.add(u"".encode('utf-8'))
banlist.add(u"".encode('utf-8'))
banlist.add(u"æ".encode('utf-8'))
banlist.add(u"".encode('utf-8'))
banlist.add(u"".encode('utf-8'))
banlist.add(u"è".encode('utf-8'))
banlist.add(u"".encode('utf-8'))
banlist.add(u"¥".encode('utf-8'))
banlist.add(u"é".encode('utf-8'))
banlist.add(u"".encode('utf-8'))
banlist.add(u"".encode('utf-8'))
banlist.add(u"æ".encode('utf-8'))
banlist.add(u"".encode('utf-8'))
banlist.add(u"".encode('utf-8'))
banlist.add(u"å".encode('utf-8'))
banlist.add(u"".encode('utf-8'))
banlist.add(u"¡".encode('utf-8'))
banlist.add(u"ä".encode('utf-8'))
banlist.add(u"¸".encode('utf-8'))
banlist.add(u"@".encode('utf-8'))
banlist.add(u"%".encode('utf-8'))
banlist.add(u"￥".encode('utf-8'))
banlist.add(u"……".encode('utf-8'))
banlist.add(u"&".encode('utf-8'))
banlist.add(u"*".encode('utf-8'))
banlist.add(u"3626".encode('utf-8'))
banlist.add(u";".encode('utf-8'))
banlist.add(u"&".encode('utf-8'))
banlist.add(u"3662".encode('utf-8'))
banlist.add(u"bbs".encode('utf-8'))
banlist.add(u".".encode('utf-8'))
banlist.add(u"e".encode('utf-8'))
banlist.add(u"5%9".encode('utf-8'))
banlist.add(u"c".encode('utf-8'))
banlist.add(u"%".encode('utf-8'))
banlist.add(u"a".encode('utf-8'))
banlist.add(u"8".encode('utf-8'))
banlist.add(u"e".encode('utf-8'))
banlist.add(u"bb".encode('utf-8'))
banlist.add(u"%88%".encode('utf-8'))
banlist.add(u"7%".encode('utf-8'))
banlist.add(u"ab".encode('utf-8'))
banlist.add(u"%".encode('utf-8'))
banlist.add(u"af".encode('utf-8'))
banlist.add(u"6%9".encode('utf-8'))
banlist.add(u"c".encode('utf-8'))
banlist.add(u"ba".encode('utf-8'))
banlist.add(u"4%".encode('utf-8'))
banlist.add(u"b".encode('utf-8'))
banlist.add(u"8%8".encode('utf-8'))
banlist.add(u"a".encode('utf-8'))
banlist.add(u"ef".encode('utf-8'))
banlist.add(u"bc".encode('utf-8'))
banlist.add(u"%8".encode('utf-8'))
banlist.add(u"6%98%".encode('utf-8'))
banlist.add(u"be".encode('utf-8'))
banlist.add(u"7%".encode('utf-8'))
banlist.add(u"a".encode('utf-8'))
banlist.add(u"6%96%87%".encode('utf-8'))
banlist.add(u"%89%".encode('utf-8'))
banlist.add(u"4%9".encode('utf-8'))
banlist.add(u"a".encode('utf-8'))
banlist.add(u"7%8".encode('utf-8'))
banlist.add(u"8%".encode('utf-8'))
banlist.add(u"6%96%87%".encode('utf-8'))
banlist.add(u"ad".encode('utf-8'))
banlist.add(u"%97.".encode('utf-8'))
banlist.add(u"doc".encode('utf-8'))



#banlist.add("#")
#banlist.add("?")

def uniq(in_list):
  last = None
  new_list = []
  for item in in_list:
    if item == last:
      continue
    else:
      new_list.append(item)
      last = item
  return new_list

def load_words(filename, words=None):
  if words is None:
    words = []
  with open(filename) as fd:
    for line in fd:
      start = 0
      if line[0] == '[':
        start=1
      end = -1
      if line[-1] == ']':
        end = -1
      if line[-2] == ']' and line[-1] == '\n':
        end = -2
      if line[-1] == '\t' and line[-2] == '\r' and line[-3] == ']':
        end = -3
      sentence = filter(lambda x: x != '', map(string.strip, line[start:end].split(',')))
      sentence = [ word for word in sentence if word not in banlist ]
      sentence = uniq(sentence)
      words.append(sentence)
  return words


class WeiboReader():
  """
    Read data from train data.
  """
  # 20% for validation.
  VADATION_RATION = 0.2 
  def __init__(self):
    self._total_uids = 0
    self._total_mids = 0
    self._total_forward_count = 0
    self._total_comment_count = 0
    self._total_like_count = 0
    self._total_zero_forward_count = 0
    self._total_zero_comment_count = 0
    self._total_zero_like_count = 0
    self._max_forward_count = 0
    self._max_forward_content = ""
    self._max_comment_count = 0
    self._max_comment_content = ""
    self._max_like_count = 0
    self._max_like_content = ""
    self._uid_avg_statistics = {}
    self._data = []
    self._uid_data = {}
    self._wid_set = Set()
    self._is_prediction_data = False
    self._training_set = None
    self._validation_set = None
    self._is_include_words = None

  def data(self):
    return self._data
    
  def _add_info(self, info):
    
    self._data.append(info)
    
    self._wid_set.add(info[1])
    self._total_mids = len(self._wid_set)

    if len(info) > 4:    
      forward_count = int(info[3])
      comment_count = int(info[4])
      like_count = int(info[5])
      self._total_forward_count += forward_count
      self._total_comment_count += comment_count
      self._total_like_count += like_count
      
      if int(info[3]) > self._max_forward_count:
        self._max_forward_count = int(info[3])
        self._max_forward_content = info
        
      if int(info[3]) <= 0:
        self._total_zero_forward_count += 1
      
      if int(info[4]) > self._max_comment_count:
        self._max_comment_count = int(info[4])
        self._max_comment_content = info
        
      if int(info[4]) <= 0:
        self._total_zero_comment_count += 1
        
      if int(info[5]) > self._max_like_count:
        self._max_like_count = int(info[5])
        self._max_like_content = info
        
      if int(info[5]) <=0:
        self._total_zero_like_count += 1
        
    if self._uid_data.has_key(info[0]):
      self._uid_data[info[0]].append(info)
    else:
      self._uid_data[info[0]] = [info]
      self._total_uids += 1

  def _caclulate_uid_info(self, all_uid_mids, is_str = False):
    if len(all_uid_mids) <= 0:
      if is_str:
        return ""
      else:
        return ()
    total_mid = len(all_uid_mids)
    total_forward = 0
    total_comment = 0
    total_like = 0
    total_zero_forward = 0
    total_zero_comment = 0
    total_zero_like = 0
    for mid_info in all_uid_mids:
      total_forward += int(mid_info[3])
      total_comment += int(mid_info[4])
      total_like += int(mid_info[5])
      if int(mid_info[3]) <= 0:
        total_zero_forward += 1
      if int(mid_info[4]) <= 0:
        total_zero_comment += 1
      if int(mid_info[5]) <= 0:
        total_zero_like += 1
        
    avg_forward = 1.0 * total_forward / total_mid
    avg_comment = 1.0 * total_comment / total_mid
    avg_like = 1.0 * total_like / total_mid

    if is_str:
      return """total mids: %d, total forward: %d, total comment: %d, 
              total like: %d, total zero forward: %d, total zero comment: %d
              total zero like: %d, avg forward: %f, avg comment: %f
              avg like: %f""" % ( 
              total_mid, total_forward, total_comment, total_like, 
              total_zero_forward, total_zero_comment, total_zero_like,
              avg_forward, avg_comment, avg_like
              )
    else:        
      return (total_mid, total_forward, total_comment, total_like, 
              total_zero_forward, total_zero_comment, total_zero_like,
              avg_forward, avg_comment, avg_like)

  def _split_train_validation_sets(self):
    if self._training_set == None:
      size = len(self._data)
      sorted_data = sorted(self._data, key=get_date_info)
      train_size = int(size * (1 - self.VADATION_RATION))
      self._training_set = self._data[0:train_size]
      self._validation_set = self._data[train_size:-1]
      return True
    else:
      return True                
        
  def get_training_data(self):
    self._split_train_validation_sets()
    return self._training_set
  
  def get_validation_data(self):
    self._split_train_validation_sets()
    return self._validation_set
    
  def _load_single_data(self, filename):
    """
      Load the data from file.
    """
    if not os.path.isfile(filename):
      return False
    
    for line in open(filename, 'r'):
      info = line.split("\t")
      if len(info) == 4:
        self._is_prediction_data = True
      elif len(info) < 7:
        print "Error: data is corrupted: ", line
        continue
      
      self._add_info(info)
      
  def load_data(self, filenames):
    if not isinstance(filenames, (list, tuple)):
      filenames = [filenames]
    
    for filename in filenames:
      self._load_single_data(filename)

  def _attach_words(self, words):
    if len(self._data) != len(words):
      print("Warning: the data and sentence words are not matched!")
    for info, word in zip(self._data, words):
      info.append(word)
    
    
  def _load_single_words_data(self, filename, words_filename):
    """
      Load data and mapped words to replace the content field of the data.
    """
    if not os.path.isfile(filename) or not os.path.isfile(words_filename):
      return False
    
    self._load_single_data(filename)
    words = load_words(words_filename)
    self._attach_words(words)
      
          
  def load_words_data(self, filenames, words_filenames):
    if not isinstance(filenames, (list, tuple)):
      filenames = [filenames]
      words_filenames = [words_filenames]
    if len(filenames) != len(words_filenames):
      print("Warning: the filename and words filename are not matched")
    for filename, words_filenames in zip(filenames, words_filenames):
      self._load_single_words_data(filename, words_filenames)
      
  def save_data(self, predictions, filename):
    fd = open(filename, 'w')
    
    for info, pre in zip(self._data, predictions):
      fd.write("%s %s %d,%d,%d\n" % (info[0], info[1], int(pre[0]), int(pre[1]), int(pre[2])))
      
    fd.close()
      
  def get_uid_info(self, uid, is_str = False):
    return self._caclulate_uid_info(self._uid_data[uid], is_str)
  
  def print_uid_top_info(self, uid):
    weibos = sorted(self._uid_data[uid], key=get_date_info)
    print "===================uid: ", uid, "information=============="
    for idx, weibo in enumerate(weibos):
      print "NO", idx, " : ", weibo[0:6], weibo[6]
      
    return

  @staticmethod
  def _count_limit(datas, get_int, limits, counts):
    for data in datas:
      count = get_int(data)
      for idx, limit in enumerate(limits):
        if count <= limit:
          counts[idx] += 1
          break
    
      
      
  
  def print_rank_info(self, forward_limits = None, comment_limits = None, 
                      like_limits = None):
    default_limits = [0,1,5,10,50,100,500,1000,1500, 2000,3000,5000,10000,20000,
                      50000,80000,100000, 200000]
    if forward_limits == None:
      forward_limits = default_limits
    if comment_limits == None:
      comment_limits = default_limits
    if like_limits == None:
      like_limits = default_limits
      
    # collect all the statistic infos
    forward_counts = [ 0 for v in forward_limits ]
    comment_counts = [ 0 for v in comment_limits ]
    like_counts = [ 0 for v in like_limits ]
    
    WeiboReader._count_limit(self._data, get_forward_count, 
                             forward_limits, forward_counts)
    WeiboReader._count_limit(self._data, get_comment_count, 
                             comment_limits, comment_counts)
    WeiboReader._count_limit(self._data, get_like_count, 
                             like_limits, like_counts)
    
    print "===================rank information=============="
    print "++  forward counts"
    for limit, count in zip(forward_limits, forward_counts):
      print "smaller than", limit, "total: ", count 
    print "++  comment counts"
    for limit, count in zip(comment_limits, comment_counts):
      print "smaller than", limit, "total: ", count
    print "++  like counts"
    for limit, count in zip(like_limits, like_counts):
      print "smaller than", limit, "total: ", count
        

  def print_top_list(self, top_n_info=50, top_n_show_detail_info=50, threshold_hide_general_info=10):
    # sort on forward
    forward_sorted = sorted(self._data, key=get_forward_count, reverse=True)          
    comment_sorted = sorted(self._data, key=get_comment_count, reverse=True)
    like_sorted = sorted(self._data, key=get_like_count, reverse=True)
    # statistic info for top n rank items
    top_n = top_n_info
    # top n show 
    top_n_show = top_n_show_detail_info
    threshold = threshold_hide_general_info
    forward_top_uids = {}
    comment_top_uids = {}
    like_top_uids = {}
    print "============ forward top", top_n_show
    for idx, weibo in enumerate(forward_sorted[0:top_n]):
      if idx < top_n_show:
        print "No", idx, weibo[0:6], weibo[6]
      if forward_top_uids.has_key(weibo[0]):
        forward_top_uids[weibo[0]] += 1 
      else:
        forward_top_uids[weibo[0]] = 1

    print "============ comment top", top_n_show
    for idx, weibo in enumerate(comment_sorted[0:top_n]):
      if idx < top_n_show:
        print "No", idx, weibo[0:6], weibo[6]
      if comment_top_uids.has_key(weibo[0]):
        comment_top_uids[weibo[0]] += 1 
      else:
        comment_top_uids[weibo[0]] = 1
      
    print "============ like top", top_n_show      
    for idx, weibo in enumerate(like_sorted[0:top_n]):
      if idx < top_n_show:
        print "No", idx, weibo[0:6], weibo[6]
      if like_top_uids.has_key(weibo[0]):
        like_top_uids[weibo[0]] += 1 
      else:
        like_top_uids[weibo[0]] = 1

              
    print "============ general info"
    print "++ forward top", top_n, "uid info:", "threshold: ", threshold
    for key, value in sorted(forward_top_uids.items(), key=operator.itemgetter(1), reverse=True):
      if value > threshold:
        print "ppl: ", key, " times: ", value, "\n          ", self.get_uid_info(key, is_str=True)
    print "++ comment top", top_n, "uid info:", "threshold: ", threshold
    for key, value in sorted(comment_top_uids.items(), key=operator.itemgetter(1), reverse=True):
      if value > threshold:
        print "ppl: ", key, " times: ", value, "\n          ", self.get_uid_info(key, is_str=True)
    print "++ like top", top_n, "uid info:", "threshold: ", threshold
    for key, value in sorted(like_top_uids.items(), key=operator.itemgetter(1), reverse=True):
      if value > threshold:
        print "ppl: ", key, " times: ", value, "\n          ", self.get_uid_info(key, is_str=True)
        
    print "=========== overlap on", top_n
    ppl_all_top = Set()
    for uid in forward_top_uids.keys():
      ppl_all_top.add(uid)
      
    for uid in comment_top_uids.keys():
      ppl_all_top.add(uid)
      
    for uid in like_top_uids.keys():
      ppl_all_top.add(uid)
      
    for uid in ppl_all_top:
      if forward_top_uids.has_key(uid) and comment_top_uids.has_key(uid) and like_top_uids.has_key(uid):
        if forward_top_uids[uid] > threshold or comment_top_uids[uid] > threshold or like_top_uids[uid] > threshold:
          print "ppl ", uid, "times:", forward_top_uids[uid], comment_top_uids[uid], like_top_uids[uid], "\n       ", self.get_uid_info(uid, is_str=True)
          
          
    print "========== time info"
    week_info = [(0,0,0)] * 7
    week_year_info = [(0,0,0)] * 54
    week_year_detail_info = [] #[[(0,0,0)] * 7]*54
    for idx in range(0,54):
      week_year_detail_info.append([(0,0,0)]*7)
    
    month_info = [(0,0,0)] * 13
    day_info = [(0,0,0)] * 32
    day_year_info = {}
    year_info = {}
    for info in self._data:
      t_list = info[2].split(" ")
      key_date = info[2]
      if len(t_list) > 1:
        # date and time.
        key_date = t_list[0]
        s = t_list[0].split("-")
        d = datetime.date(int(s[0]), int(s[1]), int(s[2]))
      else:
        s = info[2].split("-")
        d = datetime.date(int(s[0]), int(s[1]), int(s[2]))
        
      nums = week_info[d.weekday()]
      week_info[d.weekday()] = (nums[0]+int(info[3]), nums[1]+int(info[4]), nums[2]+int(info[5]))
      nums = week_year_info[d.isocalendar()[1]]
      week_year_info[d.isocalendar()[1]] = (nums[0]+int(info[3]), nums[1]+int(info[4]), nums[2]+int(info[5]))
      #if d.isocalendar()[1] == 1:
      #  print info
      nums = week_year_detail_info[d.isocalendar()[1]][d.weekday()]
      if d.isocalendar()[1] == 3:
        print "handle: ", d.isocalendar()[1], d.weekday()
      week_year_detail_info[d.isocalendar()[1]][d.weekday()] = (nums[0]+int(info[3]), nums[1]+int(info[4]), nums[2]+int(info[5]))
      
      nums = month_info[d.month]
      month_info[d.month] = (nums[0]+int(info[3]), nums[1]+int(info[4]), nums[2]+int(info[5]))
      nums = day_info[d.day]
      day_info[d.day] = (nums[0]+int(info[3]), nums[1]+int(info[4]), nums[2]+int(info[5]))
      if year_info.has_key(d.year):
        nums = year_info[d.year]
      else:
        nums = (0,0,0)
        year_info[d.year] = nums
      year_info[d.year] = (nums[0]+int(info[3]), nums[1]+int(info[4]), nums[2]+int(info[5]))
      if day_year_info.has_key(key_date):
        nums = day_year_info[key_date]
      else:
        nums = (0,0,0)
      day_year_info[key_date] = (nums[0]+int(info[3]), nums[1]+int(info[4]), nums[2]+int(info[5]))
        
    print "++ week info(Monday is 0 and Sunday is 6):"
    for idx, s_info in enumerate(week_info):
      if s_info == (0,0,0):
        continue
      print "week day ", idx, s_info
      
    for idx, s_info in enumerate(week_year_info):
      if s_info == (0,0,0):
        continue
      print "week ", idx, s_info
      
    for idx, s_info in enumerate(week_year_detail_info):
      for idx2, s_info2 in enumerate(s_info):
        if s_info2 == (0,0,0):
          continue
        print "week", idx, " weekday ", idx2, s_info2
      
    for idx, s_info in enumerate(month_info):
      if s_info == (0,0,0):
        continue
      print "month ", idx, s_info
      
    for idx, s_info in enumerate(day_info):
      if s_info == (0,0,0):
        continue
      print "day ", idx, s_info
      
    for idx, s_info in sorted(day_year_info.items(), key=operator.itemgetter(0)):
      if s_info == (0,0,0):
        continue
      print idx, s_info
      
    for idx, s_info in year_info.iteritems():
      if s_info == (0,0,0):
        continue
      print "year ", idx, s_info
    
    
      
  def print_dataset_info(self):
    print "================================="
    print "total lines: ", len(self._data)
    print "total uid: ", self._total_uids
    print "total mid: ", self._total_mids
    print "mid/uid: ", self._total_mids / self._total_uids
    if not self._is_prediction_data:
      print "total forward: ", self._total_forward_count
      print "total comment: ", self._total_comment_count
      print "total like: ", self._total_like_count
      print "avg forward(/mid): ", 1.0 * self._total_forward_count / self._total_mids
      print "avg forward(/uid): ", 1.0 * self._total_forward_count / self._total_uids
      print "avg comment(/mid): ", 1.0 * self._total_comment_count / self._total_mids
      print "avg comment(/uid): ", 1.0 * self._total_comment_count / self._total_uids
      print "avg like(/mid): ", 1.0 * self._total_like_count / self._total_mids
      print "avg like(/uid): ", 1.0 * self._total_like_count / self._total_uids
      print "total zero forward: ", self._total_zero_forward_count
      print "total zero comment: ", self._total_zero_comment_count
      print "total zero like: ", self._total_zero_like_count
      print "max forward number(single): ", self._max_forward_count, " content: ", self._max_forward_content[0:6], self._max_forward_content[6]
      print "max forward user info: ", self.get_uid_info(self._max_forward_content[0], True)
      print "max comment number(single): ", self._max_comment_count, " content: ", self._max_comment_content[0:6], self._max_comment_content[6]
      print "max comment user total mids: ", len(self._uid_data[self._max_comment_content[0]])
      print "max comment user info: ", self.get_uid_info(self._max_comment_content[0], True)
      print "max like number(single): ", self._max_like_count, " content: ", self._max_like_content[0:6], self._max_like_content[6]
      print "max like user total mids: ", len(self._uid_data[self._max_like_content[0]])
      print "max like user info: ", self.get_uid_info(self._max_like_content[0], True)
      

