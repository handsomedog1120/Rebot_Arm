#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.detectors.detector_factory import detector_factory
from lib.opts import opts

class ROSLiveDetector:
    def __init__(self):
        rospy.init_node('centernet_live', anonymous=True)
        self.bridge = CvBridge()
        self.sub = rospy.Subscriber('/camera/color/image_raw', Image, self.callback, queue_size=1)

        # 加载模型
        task = 'ctdet'
        dataset = 'forensic_paper'
        arch = 'resdcn_18'
        load_model = '/home/zf/ubuntu/CenterNet/exp/ctdet/forensic_paper_v1/model_best.pth'
        dummy_args = f'ctdet --arch {arch} --load_model {load_model} --dataset {dataset}'.split()
        opt = opts().init(dummy_args)
        opt.num_classes = 1
        opt.heads = {'hm': 1, 'wh': 2, 'reg': 2}
        self.detector = detector_factory[task](opt)
        print("检测器已加载，等待图像...")

    def callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(e)
            return
        raw = self.detector.run(img)
        results = raw['results'] if isinstance(raw, dict) and 'results' in raw else raw
        vis = img.copy()
        for cat_id, bboxes in results.items():
            for bbox in bboxes:
                if not isinstance(bbox, (list, tuple)) or len(bbox) < 5:
                    continue
                x1, y1, x2, y2, score = map(int, bbox[:5])
                if score < 0.3:
                    continue
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, f"paper {score:.2f}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("Detection", vis)
        cv2.waitKey(1)

if __name__ == '__main__':
    try:
        node = ROSLiveDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    cv2.destroyAllWindows()
