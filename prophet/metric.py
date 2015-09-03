# vim: set fileencoding=utf-8 :
#!/usr/bin/env python

def get_umid_key(item):
  return item[0]+item[1]
"""
  It contains the code to calculate precision for weibo competition
  Created by BlackCat@exadeep
"""

def get_counts(gt):
  if len(gt) >= 7:
    return (int(gt[3]),int(gt[4]),int(gt[5]))
  else:
    return (int(gt[2]),int(gt[3]),int(gt[4]))

def sgn(x):
  return 1 if x > 0 else 0

def count_i(f,c,l):
  total = f + c + l
  return 100 if total > 100 else total

class WeiboPrecision():

  def __init__(self):
    pass

  @staticmethod
  def precision(groundtruth, predictions):
    gt_map = {}
    for gt in groundtruth:
      gt_map[get_umid_key(gt)]=gt

    g_precision_top = 0.0
    g_precision_bt = 0.0
    for pre in predictions:
      key = get_umid_key(pre)
      if not gt_map.has_key(key):
        raise Exception("No groundtruth found")
      gt = gt_map[key]
      forward_gt, comment_gt, like_gt = get_counts(gt)
      forward_pre, comment_pre, like_pre = get_counts(pre)
      f_dev = abs(forward_pre - forward_gt) / (forward_gt + 5.0)
      c_dev = abs(comment_pre - comment_gt) / (comment_gt + 3.0)
      l_dev = abs(like_pre - like_gt) / (like_gt + 3.0)
      single_precision = 1 - 0.5 * f_dev - 0.25 * c_dev - 0.25 * l_dev
      counti = count_i(forward_gt, comment_gt, like_gt)
      g_precision_top += (counti + 1.0) * sgn(single_precision - 0.8)
      g_precision_bt += (counti + 1.0)

    return g_precision_top / g_precision_bt
