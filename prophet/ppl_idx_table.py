# vim: set fileencoding=utf-8 :
#!/usr/bin/env python

from sklearn.neighbors import NearestNeighbors
import numpy as np

def mydist(x,y):
  return abs(x[0]-y[0])

class PplIdxTable():
  def __init__(self):
    self._idx_table = {}
    self._missing = 0
    self._missing_ppl = {}
    self._ppl_standard = None
    self._unknow_ppl_standard_maps = None
    self._cached_ppl_idx = {}

  def reset_missing(self):
    self._missing = 0
    self._missing_ppl = {}
    
  def get_missing_count(self):
    return self._missing
  
  def get_missing_uniq_ppl(self):
    return len(self._missing_ppl)
        
  def get_table_idx(self):
    return len(self._idx_table)

  def create_ppl_idx(self, uid):
    if uid in self._idx_table:
      return self._idx_table[uid]
    else:
      idx = len(self._idx_table)
      self._idx_table[uid] = idx
      return idx
    
  def create_ppl_standard(self, standards):
    ppl_standrd = NearestNeighbors(n_neighbors=4, algorithm='ball_tree',
                                          metric='pyfunc', func=mydist)
    s_list = []
    #print(dir(standards))
    for uid, val in standards.iteritems():
      idx = self.get_ppl_idx(uid)
      if idx == -1:
        continue
      
      s_list.append((val[0], idx))
    
    ppl_standrd.fit(np.array(s_list, dtype='float32'))
    
    self._ppl_standard = ppl_standrd
    
  def init_unknow_ppl_standard_map(self, standards):
    self._unknow_ppl_standard_maps = standards
    self._cached_ppl_idx = {}
    
  def _get_nearest_ppl_idx(self, standard_val):
    ret = self._ppl_standard.kneighbors(np.array([standard_val[0], 0], dtype='float32'))
    
    if len(ret[0]) > 0:
      return int(ret[0][0][0])
    else:
      return -1
    
  def create_ppls_table(self, uids, f = None, standard=None):
    idx = []
    for uid in uids:
      if f is not None:
        idx.append([self.create_ppl_idx(f(uid))])
      else:
        idx.append([self.create_ppl_idx(uid)])
        
    if standard is not None:
      self.create_ppl_standard(standard)
    return idx

  def get_ppl_idx(self, uid, muid = None):
    if uid in self._idx_table:
      return self._idx_table[uid]
    else:
      self._missing += 1
      self._missing_ppl[uid] = 1
      if muid is not None:
        if muid in self._idx_table:
          return self._idx_table[muid]
      if self._unknow_ppl_standard_maps is not None:
        if uid in self._cached_ppl_idx:
          return self._cached_ppl_idx[uid]
        
        if uid in self._unknow_ppl_standard_maps:
          idx = self._get_nearest_ppl_idx(self._unknow_ppl_standard_maps[uid])
          if idx != -1:
            self._cached_ppl_idx[uid] = idx
          return idx        
      return -1
      
  def get_ppls_idx(self, uids, f = None, muid = None, unknow_ppl_standard = None):
    idxs = []
    if unknow_ppl_standard is not None:
      self.init_unknow_ppl_standard_map(unknow_ppl_standard)
      
    for uid in uids:
      if f is not None:
        idxs.append([self.get_ppl_idx(f(uid), muid)])
      else:
        idxs.append([self.get_ppl_idx(uid, muid)])
    return idxs
        

        

          