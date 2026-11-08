# -*- coding: utf-8 -*-
"""
Forensic3: 三物证(纸片/液体/六边形)联合检测数据集 — 单模型 3 类。
类别顺序刻意设为 paper=1, liquid=2, hexagon=3, 与 lib/datasets/sample/ctdet.py
里 AAGK 创新点(cls_id in [1,2] 对细长/不规则物体做各向异性高斯核)的假设一致:
纸片与液体落在 1,2, 六边形(规则形状)落在 3。
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import numpy as np
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval

from ..sample.ctdet import CTDetDataset


class Forensic3(CTDetDataset):
  num_classes = 3
  default_resolution = [512, 512]
  mean = np.array([0.40789654, 0.44719302, 0.47026115],
                  dtype=np.float32).reshape(1, 1, 3)
  std = np.array([0.28863828, 0.27408164, 0.27809835],
                 dtype=np.float32).reshape(1, 1, 3)

  def __init__(self, opt, split):
    super(Forensic3, self).__init__()

    self.opt = opt
    self.split = split

    self._data_rng = np.random.RandomState(123)
    self._eig_val = np.array([0.2141, 0.0181, 0.0034], dtype=np.float32)
    self._eig_vec = np.array([
        [-0.5875, -0.6956, 0.4134],
        [-0.5833,  0.0099, -0.8122],
        [-0.5608,  0.7183, 0.4116]
    ], dtype=np.float32)

    self.data_dir = os.path.join(opt.data_dir, 'forensic_3')

    if split == 'train':
      self.img_dir = os.path.join(self.data_dir, 'images', 'train2024')
      self.annot_path = os.path.join(
          self.data_dir, 'annotations', 'instances_train2024.json')
    else:
      self.img_dir = os.path.join(self.data_dir, 'images', 'val2024')
      self.annot_path = os.path.join(
          self.data_dir, 'annotations', 'instances_val2024.json')

    self.max_objs = 128
    self.class_name = ['__background__', 'paper', 'liquid', 'hexagon']
    self._valid_ids = [1, 2, 3]
    self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}

    print('==> initializing forensic_3 {} data.'.format(split))
    print('images dir:', self.img_dir)
    print('annot path:', self.annot_path)

    self.coco = coco.COCO(self.annot_path)
    self.images = self.coco.getImgIds()
    self.num_samples = len(self.images)

    print('Loaded {} {} samples'.format(split, self.num_samples))

  def __len__(self):
    return self.num_samples

  def convert_eval_format(self, all_bboxes):
    detections = []
    for image_id in all_bboxes:
      for cls_ind in all_bboxes[image_id]:
        category_id = self._valid_ids[cls_ind - 1]
        for bbox in all_bboxes[image_id][cls_ind]:
          bbox = list(map(float, bbox))
          x1, y1, x2, y2, score = bbox[:5]
          w = x2 - x1
          h = y2 - y1
          if w <= 0 or h <= 0:
            continue
          detections.append({
              'image_id': int(image_id),
              'category_id': int(category_id),
              'bbox': [x1, y1, w, h],
              'score': float(score)
          })
    return detections

  def save_results(self, results, save_dir):
    result_json = os.path.join(save_dir, 'results.json')
    detections = self.convert_eval_format(results)
    with open(result_json, 'w') as f:
      json.dump(detections, f)
    print('Saved results to {}'.format(result_json))

  def run_eval(self, results, save_dir):
    self.save_results(results, save_dir)
    result_json = os.path.join(save_dir, 'results.json')
    coco_dets = self.coco.loadRes(result_json)
    coco_eval = COCOeval(self.coco, coco_dets, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()