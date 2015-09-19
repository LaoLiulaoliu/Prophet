# vim: set fileencoding=utf-8 :
#!/usr/bin/env python

class PplIdxTable():
  def __init__(self):
    self._idx_table = {}
    self._missing = 0
    self._missing_ppl = {}

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
    
  def create_ppls_table(self, uids, f = None):
    idx = []
    for uid in uids:
      if f is not None:
        idx.append([self.create_ppl_idx(f(uid))])
      else:
        idx.append([self.create_ppl_idx(uid)])
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
        else:
          return -1
      else:
        return -1
      
  def get_ppls_idx(self, uids, f = None, muid = None):
    idxs = []
    for uid in uids:
      if f is not None:
        idxs.append([self.get_ppl_idx(f(uid), muid)])
      else:
        idxs.append([self.get_ppl_idx(uid, muid)])
    return idxs
        

        

          