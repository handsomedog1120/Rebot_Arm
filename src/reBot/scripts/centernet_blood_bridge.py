#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CenterNet 血迹检测桥接模块 (reBot UI 专用)
================================================================
加载 /home/zf/blood 下训练的 CenterNet (dla_34, 单类 blood) 权重，
接口与 yolov8_bridge.ForensicDetector 完全兼容：
  ForensicDetector.detect(color_bgr, depth_img=None) -> DetectionResult or None

DetectionResult 字段:
  - category: 'liquid' (血迹 -> 涂抹, action_num=1)
  - label: CenterNet 类别名 ('blood')
  - score: 置信度
  - bbox: (x1, y1, x2, y2) 像素坐标, 已裁剪到图像边界内
  - mask: bbox 填充的 uint8 二值掩模 (CenterNet 无实例分割输出)
  - centroid_px: bbox 中心像素坐标
  - contour: bbox 四角轮廓点集
  - all_boxes: 全部 ≥置信度阈值的检出框, UI 手动画面多目标画框用
"""
import os
import sys

import cv2
import numpy as np

# ────────────────────────────────────────────────────────────────
# 模型配置: 现场按需修改这里即可
# ────────────────────────────────────────────────────────────────
CENTERNET_ROOT = '/home/zf/blood/src'          # main.py/_init_paths.py/lib 所在目录
CENTERNET_MODEL_PATH = '/home/zf/blood/exp/ctdet/blood/model_best.pth'

# 训练时的数据集/网络结构 (见 exp/ctdet/blood/opt.txt)
CENTERNET_DATASET = 'blood'
CENTERNET_ARCH = 'dla_34'

# '0' 用第一张 GPU, '-1' 走 CPU
CENTERNET_GPUS = '0'

# 置信度阈值: 低于该分数的检测结果不予采信 (含 UI 显示)
DETECT_SCORE_THRESH = 0.3

# CenterNet 单类模型: cat_id(1-indexed) -> 类别名
CLASS_MAP = {
    1: 'blood',
}

# 类别名 -> 物证大类('solid'/'liquid'/'micro')
# 血液 -> liquid -> execute_action(1) 涂抹
LABEL_TO_CATEGORY = {
    'blood': 'liquid',
}


class DetectionResult(object):
    """一次检测结果，供上层控制器直接使用"""

    def __init__(self, category, label, score, bbox, mask, centroid_px, contour,
                 all_boxes=None):
        self.category = category        # 'solid' / 'liquid' / 'micro'
        self.label = label               # CenterNet 原始类别名, 如 'blood'
        self.score = score
        self.bbox = bbox                 # (x1, y1, x2, y2) float
        self.mask = mask                 # uint8 掩模, 与输入图像同尺寸 (bbox 内部填充)
        self.centroid_px = centroid_px   # (px, py) bbox 中心像素坐标
        self.contour = contour           # cv2 轮廓点集 (bbox 四角)
        # 全部检出框(含非最高分): [{"bbox": (x1,y1,x2,y2), "score": s, "label": n}, ...]
        # 自动采集只用单目标(最高分), UI 手动画面画框用全部.
        self.all_boxes = all_boxes if all_boxes is not None else [
            {"bbox": bbox, "score": score, "label": label}]


class ForensicDetector(object):
    """
    加载 /home/zf/blood 训练的 CenterNet 单类血迹检测器.
    """

    def __init__(self, score_thresh=DETECT_SCORE_THRESH):
        # CenterNet 源码 (lib) 加入 sys.path, 只在本进程内生效
        for _p in (CENTERNET_ROOT, os.path.join(CENTERNET_ROOT, 'lib')):
            if _p not in sys.path:
                sys.path.insert(0, _p)

        from opts import opts
        from detectors.detector_factory import detector_factory
        from datasets.dataset_factory import get_dataset

        self.score_thresh = float(score_thresh)
        self.class_map = CLASS_MAP

        if not os.path.exists(CENTERNET_MODEL_PATH):
            raise RuntimeError("CenterNet模型文件不存在: {}".format(CENTERNET_MODEL_PATH))

        argv = [
            'ctdet',
            '--dataset', CENTERNET_DATASET,
            '--arch', CENTERNET_ARCH,
            '--load_model', CENTERNET_MODEL_PATH,
            '--gpus', CENTERNET_GPUS,
        ]
        opt = opts().parse(argv)
        dataset = get_dataset(opt.dataset, opt.task)
        opt = opts().update_dataset_info_and_set_heads(opt, dataset)
        opt.debug = 0

        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
        Detector = detector_factory[opt.task]
        self.detector = Detector(opt)

        print("[centernet_blood_bridge] 已加载 CenterNet: dataset={} arch={} <- {} "
              "(score_thresh={:.2f})".format(
                  CENTERNET_DATASET, CENTERNET_ARCH,
                  CENTERNET_MODEL_PATH, self.score_thresh))

    def detect(self, color_bgr, depth_img=None):
        """
        对一帧彩色图像做检测, 返回得分最高且超过阈值的 DetectionResult;
        若没有任何检出高于阈值的目标, 返回 None.

        depth_img: 保留接口兼容(YOLO 版同签名); CenterNet 检测本身不消费深度.
        """
        h, w = color_bgr.shape[:2]
        ret = self.detector.run(color_bgr)
        results = ret.get('results', {})

        all_boxes = []
        best = None  # (score, cat_id, bbox)
        for cat_id, dets in results.items():
            label = self.class_map.get(int(cat_id), 'obj')
            for obj in dets:
                score = float(obj[4])
                if score < self.score_thresh:
                    continue
                # 裁剪到图像边界, 避免中心点落在画面外导致深度取样失败
                x1 = float(np.clip(obj[0], 0, w - 1))
                y1 = float(np.clip(obj[1], 0, h - 1))
                x2 = float(np.clip(obj[2], 0, w - 1))
                y2 = float(np.clip(obj[3], 0, h - 1))
                if x2 <= x1 or y2 <= y1:
                    continue

                bbox = (x1, y1, x2, y2)
                all_boxes.append({"bbox": bbox, "score": score, "label": label})
                if best is None or score > best[0]:
                    best = (score, int(cat_id), bbox)

        if best is None:
            return None

        score, cat_id, bbox = best
        label = self.class_map.get(cat_id, 'obj')
        category = LABEL_TO_CATEGORY.get(label)
        if category is None:
            print("[centernet_blood_bridge] 未知类别标签: {}，跳过该检测".format(label))
            return None

        x1, y1, x2, y2 = bbox
        x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

        # CenterNet 只输出 bbox: 掩模用 bbox 填充, 轮廓/质心由 bbox 给出
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1i:y2i, x1i:x2i] = 255
        contour = np.array([
            [[x1i, y1i]], [[x2i, y1i]], [[x2i, y2i]], [[x1i, y2i]]
        ], dtype=np.int32)

        return DetectionResult(
            category=category,
            label=label,
            score=score,
            bbox=bbox,
            mask=mask,
            centroid_px=(cx, cy),
            contour=contour,
            all_boxes=all_boxes,
        )
