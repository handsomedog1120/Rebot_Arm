#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8 检测桥接模块 (替代原 CenterNet 版本)
================================================================
用 Ultralytics YOLOv8 segment 模型做物证检测，与原 centernet_bridge.py 接口完全兼容：
  ForensicDetector.detect(color_bgr, depth_img=None) -> DetectionResult or None

YOLOv8 segment 输出 bbox (xyxy) + 实例分割多边形:
  - mask: 分割掩模缩放到原图尺寸的 uint8 二值图 (替代旧 bbox 填充方案)
  - centroid_px: 掩模最大轮廓的质心 (moments), 与原 CenterNet CDG-ATRG 语义一致
  - contour: 掩模最大轮廓点集
  - all_boxes: 全部检出框(≥conf), UI 手动画面多目标画框用

类别映射 (LABEL_TO_CATEGORY):
  blood -> liquid (涂抹, action_num=1)
  hair  -> micro  (粘取, action_num=2)
  __background__ 为训练占位类, 不映射 → 跳过
"""
import os
import sys

import cv2
import numpy as np

# ────────────────────────────────────────────────────────────────
# 模型配置: 现场按需修改这里即可
# ────────────────────────────────────────────────────────────────
# ★ YOLOv8 segment 模型路径 (blood/hair, 由 yolov8-gpu venv 训练)
YOLOV8_MODEL_PATH = '/home/zf/blood_hair_dataset/runs/segment/runs/train/blood/weights/best.pt'

# GPU: '0' 用第一张 GPU, 'cpu' 走 CPU (慢 ~10x)
YOLOV8_DEVICE = '0'

# 置信度阈值: 低于该分数的检测结果不予采信 (含 UI 显示)
YOLOV8_CONF = 0.8

# NMS IoU 阈值
YOLOV8_IOU = 0.45

# ★ YOLOv8 数据集名字 (train 时 data.yaml 里写的 classes 顺序)
#   m.names 自动加载, 这里留空让模型自带的 names 生效
YOLOV8_NAMES_OVERRIDE = None

# YOLOv8 类别名 -> 物证大类('solid'/'liquid'/'micro')
# 物理对应: blood(血液) -> liquid -> execute_action(1) 涂抹;
#           hair(头发)  -> micro  -> execute_action(2) 粘取
# '__background__' 是训练占位类, 不映射 → 检出即跳过
LABEL_TO_CATEGORY = {
    'blood': 'liquid',
    'hair': 'micro',
    # 'micro_paper': 'micro',
}


class DetectionResult(object):
    """一次检测结果，供上层控制器直接使用 (与原 centernet_bridge.DetectionResult 完全同字段)"""

    def __init__(self, category, label, score, bbox, mask, centroid_px, contour,
                 all_boxes=None):
        self.category = category        # 'solid' / 'liquid' / 'micro'
        self.label = label               # YOLO 原始类别名, 如 'blood'
        self.score = score
        self.bbox = bbox                 # (x1, y1, x2, y2) float
        self.mask = mask                 # uint8 掩模, 与输入图像同尺寸 (bbox 内部填充)
        self.centroid_px = centroid_px   # (px, py) bbox 中心像素坐标
        self.contour = contour           # cv2 轮廓点集 (bbox 四角)
        # ★ 全部检出框(含非最高分): [{"bbox": (x1,y1,x2,y2), "score": s, "label": n}, ...]
        #   自动采集只用单目标(最高分), UI 手动画面画框用全部.
        self.all_boxes = all_boxes if all_boxes is not None else [
            {"bbox": bbox, "score": score, "label": label}]


class ForensicDetector(object):
    """
    加载 YOLOv8 detect 模型, 对彩色图像做检测.
    与原 centernet_bridge.ForensicDetector 接口完全兼容.
    """

    def __init__(self, score_thresh=YOLOV8_CONF):
        from ultralytics import YOLO

        self.score_thresh = float(score_thresh)

        model_path = YOLOV8_MODEL_PATH
        if not os.path.exists(model_path):
            raise RuntimeError("YOLOv8 模型文件不存在: {}".format(model_path))

        self._model = YOLO(model_path)
        self._device = YOLOV8_DEVICE

        # 覆盖模型自带的类别名 (可选, 一般用模型自带的)
        self._names = YOLOV8_NAMES_OVERRIDE if YOLOV8_NAMES_OVERRIDE else self._model.names

        print("[yolov8_bridge] 已加载 YOLOv8 {} 模型: {} <- {} (device={})".format(
            self._model.task, model_path, 'segment', self._device))
        print("[yolov8_bridge] 模型类别: {}".format(self._names))

    def detect(self, color_bgr, depth_img=None):
        """
        对一帧彩色图像做检测, 返回得分最高、且超过阈值的 DetectionResult;
        若没有任何类别检出高于阈值的目标, 返回 None.

        depth_img: 与 color_bgr 同尺寸的深度图 (YOLOv8 detect 本身不消费深度,
                   此参数保留接口兼容; 后续若加 CDG-ATRG 区域生长可用)
        """
        h, w = color_bgr.shape[:2]

        # YOLOv8 predict: 返回 list[Results], len=1 (单帧)
        # BGR -> RGB: YOLOv8 自动处理, 直接传 BGR 也可以 (会内部转)
        results = self._model.predict(
            source=color_bgr,
            conf=self.score_thresh,
            iou=YOLOV8_IOU,
            device=self._device,
            verbose=False,
        )
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return None

        confs = r.boxes.conf.cpu().numpy().astype(float)
        xyxys = r.boxes.xyxy.cpu().numpy().astype(float)
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)

        def _name(cid):
            return (self._names.get(cid, 'obj')
                    if isinstance(self._names, dict) else self._names[cid])

        # ★ 全部检出框(模型 conf 阈值已过滤): 供 UI 多目标画框
        all_boxes = [{
            "bbox": tuple(float(v) for v in xyxys[i]),
            "score": float(confs[i]),
            "label": _name(cls_ids[i]),
        } for i in range(len(xyxys))]

        # ★ 只在"已映射物证类别"里选最高分作为单目标结果
        #   (__background__ 等占位/未知类不参与, 避免误触发自动动作)
        best_idx = -1
        for i in range(len(confs)):
            if LABEL_TO_CATEGORY.get(_name(cls_ids[i])) is not None:
                if best_idx < 0 or confs[i] > confs[best_idx]:
                    best_idx = i
        if best_idx < 0:
            return None

        score = float(confs[best_idx])
        xyxy = xyxys[best_idx]
        cls_id = int(cls_ids[best_idx])
        x1, y1, x2, y2 = xyxy
        bbox = (float(x1), float(y1), float(x2), float(y2))

        # 类别映射
        label = _name(cls_id)
        category = LABEL_TO_CATEGORY[label]

        # ★ segment 模型: 用真实实例分割掩模 (缩放到原图尺寸)
        #   掩模 → 最大轮廓 → moments 质心 (与原 CenterNet CDG-ATRG 语义一致);
        #   掩模缺失/为空时回退 bbox 填充 (与 detect 版行为一致, 保底可用)
        mask = np.zeros((h, w), dtype=np.uint8)
        x1i, y1i, x2i, y2i = int(max(0, x1)), int(max(0, y1)), int(min(w, x2)), int(min(h, y2))
        if x2i > x1i and y2i > y1i:
            mask[y1i:y2i, x1i:x2i] = 255   # 保底: bbox 填充

        if r.masks is not None and best_idx < len(r.masks.data):
            _m = r.masks.data[best_idx].cpu().numpy()
            if _m.max() > 0:
                _mr = cv2.resize(_m, (w, h), interpolation=cv2.INTER_NEAREST)
                mask = np.ascontiguousarray(
                    np.where(_mr > 0, 255, 0).astype(np.uint8))

        # centroid: 掩模最大轮廓质心, 回退 bbox 中心
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        contour = np.array([
            [[x1i, y1i]], [[x2i, y1i]], [[x2i, y2i]], [[x1i, y2i]]
        ], dtype=np.int32)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

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
