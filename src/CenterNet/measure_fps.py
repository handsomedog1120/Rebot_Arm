# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import _init_paths
import os
import time
import cv2

from opts import opts
from datasets.dataset_factory import get_dataset
from detectors.detector_factory import detector_factory

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
    imgs = []

    for p in image_names:
        img = cv2.imread(p)
        if img is not None:
            imgs.append(img)

    if len(imgs) == 0:
        print("No valid images.")
        return

    print("测试图片数量 = {}".format(len(imgs)))

    # 预热，避免第一张图加载模型、CUDA初始化影响FPS
    warmup_num = min(5, len(imgs))
    for img in imgs[:warmup_num]:
        detector.run(img)

    t0 = time.time()

    for img in imgs:
        detector.run(img)

    t1 = time.time()

    total_time = t1 - t0
    fps = len(imgs) / total_time if total_time > 0 else 0
    avg_time = total_time / len(imgs) if len(imgs) > 0 else 0

    print("总耗时 = {:.4f} s".format(total_time))
    print("平均单张耗时 = {:.4f} s".format(avg_time))
    print("FPS = {:.2f}".format(fps))

if __name__ == '__main__':
    main()
