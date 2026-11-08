#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CenterNet 检测桥接模块
================================================================
对应专利技术交底书 步骤S3（SPME-CenterNet 条向金字塔微物证增强检测）
+ 步骤S4前半段（CDG-ATRG 色深梯度协同自适应阈值区域生长）。

★ 按最新确认的方案：只加载【一个】联合多类别 CenterNet 检测器，
  物证类别（固体/液体/微小）由该网络在训练阶段已经绑定好的类别标签
  (1=fixed_box, 2=fixed_top, 3=micro_paper, 4=liquid_drawing，
   定义见 CenterNet/src/lib/datasets/dataset/forensic.py) 直接输出，
  不再额外跑三个单类检测器分别判断。

设计原则：不修改 CenterNet 任何源码，只是加载已经训练好的联合模型，
并直接复用 CenterNet/src/demo_adaptive.py 里已经实现好的
apply_adaptive_region_growing()（即CDG-ATRG的工程实现）做精细分割。
"""
import os
import sys

import cv2

# ────────────────────────────────────────────────────────────────
# 路径 & 模型配置：现场按需修改这里即可，不需要改动检测逻辑代码
# ★ 已对齐 forensic_ws 已验证可用的 V4 模型 (3类: tie/liquid/hexagon,
#   arch=resdcnstrip_18 条向金字塔, 训练 140epoch, val_loss=0.478)。
#   forensic_ws CenterNet 布局: main.py/demo_adaptive.py 在 root 下,
#   lib/ 也在 root/lib (没有额外的 src 层), 故 CENTERNET_SRC = root。
# ────────────────────────────────────────────────────────────────
_CENTERNET_ROOT_DEFAULT = '/home/zf/forensic_ws/src/CenterNet'
CENTERNET_ROOT = os.environ.get('CENTERNET_ROOT', _CENTERNET_ROOT_DEFAULT)
CENTERNET_SRC = CENTERNET_ROOT                    # forensic_ws 布局: 源码直接在 root 下
CENTERNET_EXP = os.path.join(CENTERNET_ROOT, 'exp', 'ctdet')

# ★ 联合多类别检测器：直接指向 forensic_ws 已固化的权重副本
#   (~/forensic_ws/models/forensic3_v4_model_best.pth, 106MB, 实测可加载)
CENTERNET_DATASET = 'forensic_3_v4'
CENTERNET_ARCH = 'resdcnstrip_18'
CENTERNET_MODEL_PATH = '/home/zf/forensic_ws/models/forensic3_v4_model_best.pth'

# 现场若是 CPU-only 环境，把 '0' 改成 '-1'
CENTERNET_GPUS = '0'

# 置信度阈值：低于该分数的检测结果不予采信
DETECT_SCORE_THRESH = 0.3

# forensic_ws V4 3 类标签 -> 本方案内部使用的物证大类('solid'/'liquid'/'micro')。
# 物理对应(理论无误)：liquid(液体)->涂抹；hexagon(六边形固体)->夹取；tie(丝带/纸条微小)->粘取。
# forensic_ws 的 get_class_map('forensic_3_v4') 返回空, 故类别名由本地 CLASS_MAP 提供。
LABEL_TO_CATEGORY = {
    'liquid': 'liquid',    # liquid_drawing 液体类 -> execute_action(1) 涂抹
    'hexagon': 'solid',    # 六边形柱(固体)       -> execute_action(3) 夹取
    'tie': 'micro',        # 丝带/纸条(微小)      -> execute_action(2) 粘取
}

# Forensic3V4 数据集训练时绑定的类别顺序 (cat_id -> 类别名)
CLASS_MAP = {
    1: 'tie',
    2: 'liquid',
    3: 'hexagon',
}


def _ensure_centernet_on_path():
    if CENTERNET_SRC not in sys.path:
        sys.path.insert(0, CENTERNET_SRC)
    # demo_adaptive.py 自己会 `import _init_paths` 把 lib 目录加入 sys.path，
    # 这里额外保险起见提前加一次，避免部分调用顺序下 lib 还没注册
    lib_path = os.path.join(CENTERNET_SRC, 'lib')
    if lib_path not in sys.path:
        sys.path.insert(0, lib_path)


class DetectionResult(object):
    """一次检测+精细分割的结果，供上层控制器直接使用"""

    def __init__(self, category, label, score, bbox, mask, centroid_px, contour):
        self.category = category        # 'solid' / 'liquid' / 'micro'
        self.label = label               # CenterNet原始类别名，如 'fixed_box'
        self.score = score
        self.bbox = bbox                 # (x1,y1,x2,y2)
        self.mask = mask                 # uint8 掩模，与输入图像同尺寸
        self.centroid_px = centroid_px   # (px, py) 精细分割质心像素坐标
        self.contour = contour           # cv2 轮廓点集


class ForensicDetector(object):
    """
    加载一个联合多类别 CenterNet 检测器（SPME-CenterNet对应的训练权重），
    对彩色图像做检测，并对最高分检测框做CDG-ATRG精细分割。
    """

    def __init__(self, score_thresh=DETECT_SCORE_THRESH):
        _ensure_centernet_on_path()

        from opts import opts
        from detectors.detector_factory import detector_factory
        from datasets.dataset_factory import get_dataset
        import demo_adaptive as cn_demo  # 复用 apply_adaptive_region_growing 等

        self.score_thresh = score_thresh
        self._cn_demo = cn_demo

        model_path = CENTERNET_MODEL_PATH
        if not os.path.exists(model_path):
            raise RuntimeError("CenterNet模型文件不存在: {}".format(model_path))

        argv = [
            'ctdet',
            '--dataset', CENTERNET_DATASET,
            '--arch', CENTERNET_ARCH,
            '--load_model', model_path,
            '--gpus', CENTERNET_GPUS,
        ]
        opt = opts().parse(argv)
        dataset = get_dataset(opt.dataset, opt.task)
        opt = opts().update_dataset_info_and_set_heads(opt, dataset)
        opt.debug = 0

        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
        Detector = detector_factory[opt.task]
        self.detector = Detector(opt)

        # cat_id(1-indexed) -> 类别名。forensic_ws 的 get_class_map('forensic_3_v4')
        # 返回空 (forensic_ws 的 v4 检测节点另用自带 CLASS_NAMES 常量)，故本地提供该映射。
        self.class_map = CLASS_MAP

        print("[centernet_bridge] 已加载联合检测器: dataset={} arch={} <- {}".format(
            CENTERNET_DATASET, CENTERNET_ARCH, model_path))

    def detect(self, color_bgr, depth_img=None):
        """
        对一帧彩色图像做检测，返回得分最高、且超过阈值的 DetectionResult；
        若没有任何类别检出高于阈值的目标，返回 None。
        depth_img: 与 color_bgr 同尺寸的深度图（可为None，此时区域生长退化为
                   纯颜色约束，对应CDG-ATRG里"深度约束不可用时的兜底"）
        """
        ret = self.detector.run(color_bgr)
        results = ret.get('results', {})

        best = None  # (score, cat_id, bbox)
        for cat_id, dets in results.items():
            for obj in dets:
                score = float(obj[4])
                if score < self.score_thresh:
                    continue
                if best is None or score > best[0]:
                    best = (score, cat_id, obj[:4])

        if best is None:
            return None

        score, cat_id, bbox = best
        label = self.class_map.get(cat_id, 'obj')
        category = LABEL_TO_CATEGORY.get(label)
        if category is None:
            print("[centernet_bridge] 未知类别标签: {}，跳过该检测".format(label))
            return None

        x1, y1, x2, y2 = bbox
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

        mask = self._cn_demo.apply_adaptive_region_growing(
            color_bgr.copy(), (cx, cy), bbox=bbox, depth_img=depth_img
        )

        contour = None
        centroid_px = (cx, cy)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(contour)
            if M['m00'] != 0:
                centroid_px = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

        return DetectionResult(
            category=category,
            label=label,
            score=score,
            bbox=bbox,
            mask=mask,
            centroid_px=centroid_px,
            contour=contour,
        )
