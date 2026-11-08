# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import _init_paths
import os
import time
import cv2
import numpy as np

from opts import opts
from datasets.dataset_factory import get_dataset
from detectors.detector_factory import detector_factory

# 从 demo_adaptive.py 中导入自适应区域生长函数
from demo_adaptive import load_depth_map_by_image_path
from demo_adaptive import apply_adaptive_region_growing

image_ext = ['jpg', 'jpeg', 'png', 'webp']

def collect_images(path):
    if os.path.isdir(path):
        names = []
        for f in sorted(os.listdir(path)):
            ext = f.split('.')[-1].lower()
            if ext in image_ext:
                names.append(os.path.join(path, f))
        return names
    return [path]

def main():
    opt = opts().parse()

    Dataset = get_dataset(opt.dataset, opt.task)
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    opt.debug = 0

    Detector = detector_factory[opt.task]
    detector = Detector(opt)

    image_names = collect_images(opt.demo)

    valid_images = []
    for p in image_names:
        img = cv2.imread(p)
        if img is not None:
            valid_images.append(p)

    if len(valid_images) == 0:
        print("No valid images.")
        return

    print("测试图片数量 =", len(valid_images))

    # 预热
    warmup_num = min(5, len(valid_images))
    for image_name in valid_images[:warmup_num]:
        img_raw = cv2.imread(image_name)
        ret = detector.run(image_name)
        results = ret['results']

        for cat_id in results:
            for obj in results[cat_id]:
                score = float(obj[4])
                if score <= opt.vis_thresh:
                    continue

                bbox = obj[:4]
                cx = int((bbox[0] + bbox[2]) / 2)
                cy = int((bbox[1] + bbox[3]) / 2)

                depth_img = load_depth_map_by_image_path(image_name)
                mask = apply_adaptive_region_growing(
                    img_raw.copy(),
                    (cx, cy),
                    bbox=bbox,
                    depth_img=depth_img
                )

    # 正式计时：检测 + 自适应区域生长
    t0 = time.time()

    for image_name in valid_images:
        img_raw = cv2.imread(image_name)
        ret = detector.run(image_name)
        results = ret['results']

        for cat_id in results:
            for obj in results[cat_id]:
                score = float(obj[4])
                if score <= opt.vis_thresh:
                    continue

                bbox = obj[:4]
                cx = int((bbox[0] + bbox[2]) / 2)
                cy = int((bbox[1] + bbox[3]) / 2)

                depth_img = load_depth_map_by_image_path(image_name)
                mask = apply_adaptive_region_growing(
                    img_raw.copy(),
                    (cx, cy),
                    bbox=bbox,
                    depth_img=depth_img
                )

    t1 = time.time()

    total_time = t1 - t0
    avg_time = total_time / len(valid_images)
    fps = len(valid_images) / total_time if total_time > 0 else 0

    print("总耗时 = {:.4f} s".format(total_time))
    print("平均单张耗时 = {:.4f} s".format(avg_time))
    print("FPS = {:.2f}".format(fps))

if __name__ == '__main__':
    main()
