# vim: set fileencoding=utf-8 :
#!/usr/bin/env python

from weibo_reader import WeiboReader
from ppl_idx_table import PplIdxTable
import numpy as np
from gensim.models import Word2Vec
from gensim.models import Phrases
from sklearn.neighbors import KDTree
from collections import defaultdict

def get_phrase_list(p_list, n, sen):
  if n == 0:
    return sen
  else:
    return p_list[n-1][get_phrase_list(p_list, n-1, sen)]
  
  
def find_max_seq(data, max_len, func = lambda info: info[7], is_print = False):
  ret = max_len
  for info in data:
    size = len(func(info))
    
    if size > 120 and is_print:
      print("---------------")
      for inf in func(info):
        print(inf)
      
    if ret < size:
      ret = size
  return ret

def collect_words_vec(word_model, words, max_len, vector_size):
  vec = np.zeros([max_len, vector_size], dtype='float32')
  offset = 0
  for idx, word in enumerate(words):
    if word in word_model:
      if idx - offset >= max_len:
        return vec
      vec[idx-offset,:] = word_model[word]
    else:
      offset += 1      
  return vec

def search_rank_idx(score, ranks):
  score=int(score)
  for idx, (lower, r_v, upper) in enumerate(ranks):  
    if lower <= score and upper >= score:
      return idx
  
  return score
  
class WordVectors():
  def __init__(self):
    self._word2vec = None
    self._phrases = None
    
    
  def load(self, filename, phrase_filenames = None):
    if filename is None:
      return
    self._word2vec = Word2Vec.load(filename)
    
    if phrase_filenames is not None:
      self._phrases = []
      if type(phrase_filenames) is not list:
        phrase_filenames = [phrase_filenames]
      for phrase_filename in phrase_filenames:
        self._phrases.append(Phrases.load(phrase_filename))
        
  def get_single_words_vec(self, words, max_len = None, f = None, is_conv2d = False):
    new_words_list = self.translate_to_phrases(words, f)
    if is_conv2d:
      return [collect_words_vec(self._word2vec, new_words_list[0], max_len, self._word2vec.vector_size)]
    else:
      return collect_words_vec(self._word2vec, new_words_list[0], max_len, self._word2vec.vector_size)
  
  def get_words_vecs(self, words, max_len = None, f = None, is_conv2d = False):
    if type(words) == list and type(words[0]) == list:
      return [ self.get_single_words_vec(single_words, max_len, f, is_conv2d) for single_words in words]
    else:
      return [ self.get_single_words_vec(words, max_len, f, is_conv2d) ]
  
  def translate_to_phrases(self, words, f = None):
    """
      Args:
        words: [words] or [[words]]
      Return:
        [[words],[words]]
    """
    if f is not None:
      if type(words) == list and type(words[0]) == list:
        words = [ f(word) for word in words ]
      else:
        words = [ f(words) ]
      
    if self._phrases is None:
      return words
    
    return get_phrase_list(self._phrases, len(self._phrases) - 1, words)
  
class WeiboMatrix():
  refs = defaultdict(int)
  def __init__(self, dataset, len_func = None, shape_func=None, data_func=None):
    """
      Args:
        type: training/validation/predict
    """
    self._dataset = dataset
    self._len_func = len_func
    self._shape_func = shape_func
    self._data_func = data_func
    
  def __getitem__(self,key):
    if isinstance(key, slice):
      if key.start < 0 or key.stop > self.__len__():
        print("my error")
        raise IndexError
      else:
        return self._data_func(self._dataset, key.start, key.stop)
    elif isinstance(key, int):
      if key < 0 or key > self.__len__():
        print("my error")
        raise IndexError
      else:
        return self._data_func(self._dataset, key, key+1)
      
    elif isinstance(key, np.ndarray):
      if np.max(key) < self.__len__():
        val = []
        for idx in key.tolist():
          s_data = self._data_func(self._dataset, idx, idx+1)
          val.append(s_data[0])
        return np.array(val)
      else:
        raise IndexError
    elif isinstance(key, list): 
      if max(key) < self.__len__():
        val = []
        for idx in key:
          s_data = self._data_func(self._dataset, idx, idx+1)
          val.append(s_data[0])
        return np.array(val)
      else:
        raise IndexError
      
      
  def __len__(self):
    if self._len_func is None:
      return 0
    else:
      return self._len_func(self._dataset)
    
  @property
  def shape(self):
    if self._shape_func is None:
      return tuple([self.__len__()])
    else:
      return tuple([self.__len__(), self._shape_func(self._dataset)])
      

def l_slice_data(l_data, start, end):
  if start is None:
    start = 0
  if end is None:
    end = len(l_data)
  if start == 0 and end == len(l_data):
    return l_data
  return l_data[start:end]
  
class WeiboDataset():
  """
    It creates and convert all the data
  """
  def __init__(self, max_limit = None, ranking_func = None):
    self._train_reader = None
    self._predict_reader = None
    self._ppl_idx_table = PplIdxTable()
    self._is_init_all_tr = False
    self._words_vector = None
    self._max_len = None
    self._max_limit = max_limit
    if ranking_func is not None:
      self._f_ranks = ranking_func(1000000, 0.2, 5)
      self._f_ranks_np = np.array([ pred for l, pred, u in self._f_ranks])
      self._c_ranks = ranking_func(1000000, 0.2, 3)
      self._c_ranks_np = np.array([ pred for l, pred, u in self._c_ranks])
      self._l_ranks = ranking_func(1000000, 0.2, 3)
      self._l_ranks_np = np.array([ pred for l, pred, u in self._l_ranks])
    else:
      self._f_ranks = None
      self._c_ranks = None
      self._l_ranks = None
      
    
  def max_len(self):
    return self._max_len
  
  def get_training_len(self):
    if self._train_reader is None:
      return 0
    
    if self._is_init_all_tr:
      return len(self._train_reader.data())
    else:
      return len(self._train_reader.get_training_data())
    
  def get_validation_len(self):
    if self._train_reader is None:
      return 0
    if self._is_init_all_tr:
      return 0
    else:
      return len(self._train_reader.get_validation_data())
    
  def get_prediction_len(self):
    if self._predict_reader is None:
      return 0
    return len(self._predict_reader.data())
    
  def _init_ppl_table(self, is_init_all_tr=False, is_init_ppl_standard=True):
    if self._train_reader is not None:
      if is_init_all_tr:
        data = self._train_reader.data()
      else:
        data = self._train_reader.get_training_data()
        
      self._ppl_idx_table.create_ppls_table(data, lambda info: info[0])  

    if is_init_ppl_standard and self._train_reader is not None:
      val_standard = {}
      if is_init_all_tr:
        data_standard = self._train_reader.get_uid_standards("all")
      else:
        data_standard = self._train_reader.get_uid_standards("training")
        val_standard = self._train_reader.get_uid_standards("validation")
        
      if self._predict_reader is not None:
        predict_stand = self._predict_reader.get_uid_standards("all")
        val_standard.update(predict_stand)
      
      self._ppl_idx_table.create_ppl_standard(data_standard)
      self._ppl_idx_table.init_unknow_ppl_standard_map(val_standard)
      
    
  def load_data(self, train_filename, predict_filename = None, is_init_all_tr = False, is_init_ppl_standard=True):
    self._is_init_all_tr = is_init_all_tr
    if train_filename is not None:
      self._train_reader = WeiboReader(self._max_limit)
      if type(train_filename) != list:
        train_filenames = [train_filename]
      
      for train_filename in train_filenames:
        if type(train_filename) == tuple:
          self._train_reader.load_words_data(train_filename[0], train_filename[1])
        else:
          self._train_reader.load_data(train_filename)
          
      
    if predict_filename is not None:
      self._predict_reader = WeiboReader(self._max_limit)
      if type(predict_filename) != list:
        predict_filenames = [predict_filename]
        
      for predict_filename in predict_filenames:
        if type(predict_filename) == tuple:
          self._predict_reader.load_words_data(predict_filename[0], predict_filename[1])
        else:
          self._predict_reader.load_data(predict_filename)
          
    self._init_ppl_table(is_init_all_tr, is_init_ppl_standard)
      
          
  def _calculate_max_seq(self, max_len=0, is_print = False):
    
    if self._train_reader is not None:
      if self._words_vector is not None:
        words = self._words_vector.translate_to_phrases(self._train_reader.data(), lambda info: info[7])
        max_len = find_max_seq(words, max_len, lambda info: info, is_print)
      else:
        words = self._train_reader.data()
        max_len = find_max_seq(words, max_len, is_print=is_print)
        
    if self._predict_reader is not None:
      if self._words_vector is not None:
        words = self._words_vector.translate_to_phrases(self._predict_reader.data(), lambda info: info[4])
        max_len = find_max_seq(words, max_len, lambda info: info, is_print)
      else:
        max_len = find_max_seq(self._predict_reader.data(), max_len, 
                               lambda info: info[4], is_print)
        
    return max_len

  def load_words_vec(self, filename, phrase_filenames=None, max_len = None):
    if filename is not None:
      self._words_vector = WordVectors()
      self._words_vector.load(filename, phrase_filenames)
    
      if max_len != None:
        self._max_len = max_len
      else:
        self._max_len = self._calculate_max_seq()

  def get_words_vec_training_data(self, start = None, end = None, is_conv_2d=False):
    if self._words_vector is None:
      return []
    if self._train_reader is None:
      return []
    l_data = None
    if self._is_init_all_tr:
      l_data = self._train_reader.data()
               
    else:
      l_data = self._train_reader.get_training_data()

    l_data = l_slice_data(l_data, start, end)
    
    return self._words_vector.get_words_vecs(
                  l_data, 
                  self._max_len, 
                  lambda info: info[7],
                  is_conv_2d
                  ) 
             
  def get_words_vec_training_data_np(self, start = None, end = None, is_conv_2d=False):
    return np.array(self.get_words_vec_training_data(start, end, is_conv_2d), dtype='float32')
  
  def get_words_vec_training_data_matrix(self, is_conv_2d = False):
    return WeiboMatrix(self, 
                       lambda d: d.get_training_len(), 
                       lambda d: d.get_words_vec_training_data_np(0,1, is_conv_2d).shape,
                       lambda d, s, e: d.get_words_vec_training_data_np(s, e, is_conv_2d)  
                       )
  
  def get_words_vec_validation_data(self, start = None, end = None, is_conv_2d=False):
    if self._words_vector is None:
      return []
    if self._train_reader is None:
      return []
    if self._is_init_all_tr:
      return []
    l_data = self._train_reader.get_validation_data()
    l_data = l_slice_data(l_data, start, end)
    return self._words_vector.get_words_vecs(
                  l_data, 
                  self._max_len, 
                  lambda info: info[7],
                  is_conv_2d
                  )
    
  def get_words_vec_validation_data_np(self, start=None, end=None, is_conv_2d=False):
    return np.array(self.get_words_vec_validation_data(start, end, is_conv_2d), dtype='float32')

  def get_words_vec_validation_data_matrix(self, is_conv_2d=False):  
    return WeiboMatrix(self, 
                       lambda d: d.get_validation_len(), 
                       lambda d: d.get_words_vec_validation_data_np(0,1, is_conv_2d).shape,
                       lambda d, s, e: d.get_words_vec_validation_data_np(s, e, is_conv_2d)  
                       )
  
  def get_words_vec_predict_data(self, start=None, end=None, is_conv_2d=False):
    if self._words_vector is None:
      return []
    if self._predict_reader is None:
      return []
    l_data = self._predict_reader.data()
    l_data = l_slice_data(l_data, start, end)
    return self._words_vector.get_words_vecs(l_data, self._max_len, 
                                             lambda info: info[4], is_conv_2d)
  
  def get_words_vec_predict_data_np(self, start=None, end=None, is_conv_2d=False):
    return np.array(self.get_words_vec_predict_data(start, end, is_conv_2d), dtype='float32')
  
  def get_words_vec_predict_data_matrix(self, is_conv_2d=False):  
    return WeiboMatrix(self, 
                       lambda d: d.get_prediction_len(), 
                       lambda d: d.get_words_vec_predict_data_np(0,1, is_conv_2d).shape,
                       lambda d, s, e: d.get_words_vec_predict_data_np(s, e, is_conv_2d)  
                       )
  
    
  def get_ppl_training_data(self, start=None, end=None):
    if self._train_reader is None:
      return []
    if self._is_init_all_tr:
      l_data = self._train_reader.data()
    else:
      l_data = self._train_reader.get_training_data()
      
    l_data = l_slice_data(l_data, start, end)
    
    return self._ppl_idx_table.get_ppls_idx(l_data, lambda info: info[0])

  def get_ppl_training_data_np(self, start=None, end=None):
    return np.array( self.get_ppl_training_data(start, end), dtype='int' )

  def get_ppl_training_data_matrix(self):  
    return WeiboMatrix(self, 
                       lambda d: d.get_training_len(), 
                       lambda d: d.get_ppl_training_data_np(0,1).shape,
                       lambda d, s, e: d.get_ppl_training_data_np(s, e)  
                       )
  
  def get_ppl_validation_data(self, start=None, end=None):
    if self._train_reader is None or self._is_init_all_tr:
      return []
    return self._ppl_idx_table.get_ppls_idx(
                        l_slice_data(self._train_reader.get_validation_data(), start, end), 
                        lambda info: info[0]
                        )
  
  def get_ppl_validation_data_np(self, start=None, end=None):
    return np.array(self.get_ppl_validation_data(), dtype='int')
  
  def get_ppl_validation_data_matrix(self):  
    return WeiboMatrix(self, 
                       lambda d: d.get_validation_len(), 
                       lambda d: d.get_ppl_validation_data_np(0,1).shape,
                       lambda d, s, e: d.get_ppl_validation_data_np(s, e)  
                       )
  
  def get_ppl_predict_data(self, start=None, end=None):
    if self._predict_reader is None:
      return []
    return self._ppl_idx_table.get_ppls_idx(
                        l_slice_data(self._predict_reader.data(), start, end), 
                        lambda info: info[0]
                        )
  
  def get_ppl_predict_data_np(self, start=None, end=None):
    return np.array(self.get_ppl_predict_data(start, end), dtype='int')
  
  def get_ppl_predict_data_matrix(self):  
    return WeiboMatrix(self, 
                       lambda d: d.get_prediction_len(), 
                       lambda d: d.get_ppl_predict_data_np(0,1).shape,
                       lambda d, s, e: d.get_ppl_predict_data_np(s, e)  
                       )
  
  def get_training_data_gt(self, start=None, end=None, is_ranking=False):
    if self._train_reader is None:
      return []
    if self._is_init_all_tr:
      return [ self.get_ranked_value(info[3], info[4], info[5], is_ranking)  
              for info in l_slice_data(self._train_reader.data(), start, end)]
    else:
      return [ self.get_ranked_value(info[3], info[4], info[5], is_ranking) 
               for info in l_slice_data(self._train_reader.get_training_data()
                                        , start, end)]
  
  def get_training_data_gt_np(self, start=None, end=None, is_ranking=False):
    return np.array(self.get_training_data_gt(start, end, is_ranking), 
                    dtype='float32')
  
  def get_training_data_gt_matrix(self, is_ranking=False):  
    return WeiboMatrix(self, 
                       lambda d: d.get_training_len(), 
                       lambda d: d.get_training_data_gt_np(0,1).shape,
                       lambda d, s, e: d.get_training_data_gt_np(s, e, is_ranking)  
                       )

  def translate_ranked_value(self, idx, type="forward"):
    if type == "forward" and self._f_ranks is not None:
      return self._f_ranks[idx][1]
    if type == "comment" and self._c_ranks is not None:
      return self._c_ranks[idx][1]
    if type == "like" and self._l_ranks is not None:
      return self._l_ranks[idx][1]
  
  def translate_ranking(self, ranking):
    #print(ranking, len(self._f_ranks_np))
    if ranking.dtype != 'int32':
      ranking = np.array(np.rint(ranking), dtype='int32') 
      
    f = ranking[:,0]
    f[f>=len(self._f_ranks_np)] = len(self._f_ranks_np)-1
    f[f<0] = 0
    #print(f)
    c = ranking[:,1]
    c[c>=len(self._c_ranks_np)] = len(self._c_ranks_np)-1
    c[c<0] = 0
    l = ranking[:,2]
    l[l>=len(self._l_ranks_np)] = len(self._l_ranks_np)-1
    l[l<0] = 0
    if self._f_ranks_np is not None:
      #print(self._f_ranks)
      #print(self._f_ranks_np)
      #print(f[0:10])
      f = self._f_ranks_np[f]
      #print(f[0:10])
    if self._c_ranks_np is not None:
      c = self._c_ranks_np[c]
    if self._l_ranks_np is not None:
      l = self._l_ranks_np[l]
    
    return np.array([f,c,l]).transpose()
    
  
  def get_ranked_value(self, f, c, l, is_ranking):
    if not is_ranking:
      return [f, c, l]
    
    if self._f_ranks is not None:
      f = search_rank_idx(f, self._f_ranks)
    if self._c_ranks is not None:
      c = search_rank_idx(c, self._c_ranks)
    if self._l_ranks is not None:
      l = search_rank_idx(l, self._l_ranks)
    return [f, c, l]

  @staticmethod
  def _get_weighted_metric(ranks):
    l = []
    for lower, pre, upper in ranks:
      weights = 1
      if pre > 100:
        weights = 101
      else:
        weights = pre + 1
      l.append(weights)
    return l
  
  def get_ranked_weighted_metric(self):
    f = []
    if self._f_ranks is not None:
      f = WeiboDataset._get_weighted_metric(self._f_ranks)
    c = []
    if self._c_ranks is not None:
      c = WeiboDataset._get_weighted_metric(self._c_ranks)
    l = []
    if self._l_ranks is not None:
      l = WeiboDataset._get_weighted_metric(self._l_ranks)
    return (f, c, l)
  
  def get_ranked_weighted_metric_np(self):
    f, c, l = self.get_ranked_weighted_metric()
    return (np.array(f, dtype='int32'), np.array(c, dtype='int32'), np.array(l, dtype='int32')) 
    
  def get_validation_data_gt(self, start=None, end=None, is_ranking=False):
    if self._train_reader is None:
      return []
    return [ self.get_ranked_value(info[3], info[4], info[5], is_ranking) 
             for info in l_slice_data(self._train_reader.get_validation_data(),
                                       start, end)]
  
  def get_validation_data_gt_np(self, start=None, end=None, is_ranking=False):
    return np.array(self.get_validation_data_gt(start, end, is_ranking), dtype='float32')
  
  def get_validation_data_gt_matrix(self, is_ranking=False):  
    return WeiboMatrix(self, 
                       lambda d: d.get_validation_len(), 
                       lambda d: d.get_validation_data_gt_np(0,1).shape,
                       lambda d, s, e: d.get_validation_data_gt_np(s, e, is_ranking)  
                       )
  
  def get_missing_info(self, is_valid = True, is_predict = True, is_max_len = True, 
                       is_print=False):
    missing={}
    if is_valid and self._train_reader is not None:
      self._ppl_idx_table.reset_missing()
      self.get_ppl_validation_data()
      missing['val'] = self._ppl_idx_table.get_missing_uniq_ppl()
      missing['val_c'] = self._ppl_idx_table.get_missing_count()
      
    if is_predict and self._predict_reader is not None:
      self._ppl_idx_table.reset_missing()
      self.get_ppl_predict_data()
      missing['pre'] = self._ppl_idx_table.get_missing_uniq_ppl()
      missing['pre_c'] = self._ppl_idx_table.get_missing_count()
    if is_max_len:
      missing['max_len'] = self._calculate_max_seq(is_print=is_print)
      missing['set_max_len'] = self._max_len
    return missing
      
      
  def get_ppl_max_count(self):
    if self._ppl_idx_table is None:
      return 0
    else:
      return self._ppl_idx_table.get_table_idx()
    
  def save_predictions(self, pre, filename):
    if self._predict_reader is None:
      return False
    else:
      self._predict_reader.save_data(pre, filename)
      
  
      
      
  
   
  
  