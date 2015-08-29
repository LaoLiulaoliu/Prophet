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
  s = item[0].split("-")
  return datetime.date(s[0], s[1], s[2])

class WeiboReader():
  """
    Read data from train data.
  """
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
        
        
    
  def load_data(self, filename):
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
      
  def get_uid_info(self, uid, is_str = False):
    return self._caclulate_uid_info(self._uid_data[uid], is_str)

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
      if day_year_info.has_key(info[2]):
        nums = day_year_info[info[2]]
      else:
        nums = (0,0,0)
      day_year_info[info[2]] = (nums[0]+int(info[3]), nums[1]+int(info[4]), nums[2]+int(info[5]))
        
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
      

