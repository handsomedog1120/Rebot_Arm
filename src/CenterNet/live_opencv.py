#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('TkAgg')

import cv2
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.detectors.detector_factory import detector_factory
from lib.opts import opts

class LiveDetector:
    def __init__(self):
        self.task = 'ctdet'
        self.dataset = 'forensic_paper'
        self.arch = 'resdcn_18'
        self.load_model = '/home/zf/ubuntu/CenterNet/exp/ctdet/forensic_paper_v1/model_best.pth'

        dummy_args = 'ctdet --arch {} --load_model {} --dataset {}'.format(
            self.arch, self.load_model, self.dataset).split()
        opt = opts().init(dummy_args)
        opt.num_classes = 1
        opt.heads = {'hm': 1, 'wh': 2, 'reg': 2}
        self.detector = detector_factory[self.task](opt)

        for dev in [0, 1, 2]:
            self.cap = cv2.VideoCapture(dev)
            if self.cap.isOpened():
                print(f"成功打开相机 /dev/video{dev}")
                break
        else:
            print("无法打开相机")
            sys.exit(1)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("实时检测已启动，窗口将显示检测结果。按 Ctrl+C 在终端退出。")

        self.fig, self.ax = plt.subplots()
        self.im = self.ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
        self.ax.axis('off')
        self.fig.canvas.manager.set_window_title('CenterNet Real-time Detection')
        plt.tight_layout()

        self.ani = FuncAnimation(self.fig, self.update_frame, interval=50, blit=False)
        plt.show()

    def update_frame(self, frame_idx):
        ret, frame = self.cap.read()
        if not ret:
            return

        raw = self.detector.run(frame)
        results = raw['results'] if isinstance(raw, dict) and 'results' in raw else raw

        vis_img = frame.copy()
        if isinstance(results, dict):
            for cat_id, bboxes in results.items():
                if not isinstance(bboxes, list):
                    continue
                for bbox in bboxes:
                    if not isinstance(bbox, (list, tuple)) or len(bbox) < 5:
                        continue
                    x1, y1, x2, y2, score = map(int, bbox[:5])
                    if score < 0.3:
                        continue
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(vis_img, f"paper {score:.2f}", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        vis_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
        self.im.set_data(vis_img_rgb)
        self.ax.set_title(f"Frame {frame_idx}")

if __name__ == '__main__':
    detector = LiveDetector()
