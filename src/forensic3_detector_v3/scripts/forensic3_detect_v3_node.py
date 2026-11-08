#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
forensic3_detect_v3_node.py
---------------------------
ROS node: 实时 CenterNet 3 类物证检测 (tie / liquid / hexagon) on Orbbec Astra.
v3 配套模型 forensic_3_v3 (数据源 ev1/ev2/ev3, tie 重新采集扩充, 从头训练).
类别顺序 tie=1, liquid=2, hexagon=3 (与 v2 一致, AAGK cls_id in [1,2] 各向异性核
作用于 tie/liquid 细长不规则物). 与 v1/v2 仅话题命名空间 + 模型路径不同, 可共存.

运行于 conda py3.8 + PyTorch (centernet_4060), 由 forensic3_detect_v3 bash 包装启动,
绕开 ROS Melodic 的 python2. 不用 cv_bridge, 手动 sensor_msgs/Image<->numpy.
"""
import os
import sys
import time

import cv2
import numpy as np

import rospy
import std_msgs.msg
from sensor_msgs.msg import Image

CENTERNET_SRC = os.environ.get(
    "CENTERNET_SRC", "/home/zf/forensic_ws/src/CenterNet")
if CENTERNET_SRC not in sys.path:
    sys.path.insert(0, CENTERNET_SRC)
if os.path.join(CENTERNET_SRC, "lib") not in sys.path:
    sys.path.insert(0, os.path.join(CENTERNET_SRC, "lib"))

from lib.opts import opts                       # noqa: E402
from lib.detectors.detector_factory import detector_factory   # noqa: E402


CLASS_NAMES = {1: "tie", 2: "liquid", 3: "hexagon"}
CLASS_COLORS = {1: (255, 255, 0), 2: (0, 165, 255), 3: (0, 255, 0)}


def image_msg_to_bgr(msg):
    h, w = int(msg.height), int(msg.width)
    data = np.frombuffer(msg.data, dtype=np.uint8)
    enc = (msg.encoding or "").lower()
    if enc in ("rgb8",):
        return cv2.cvtColor(data.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
    if enc in ("bgr8",):
        return data.reshape(h, w, 3)
    if enc in ("mono8", "8uc1", "type_8uc1"):
        return cv2.cvtColor(data.reshape(h, w), cv2.COLOR_GRAY2BGR)
    ch = int(msg.step) // w if w else 0
    if ch == 1:
        return cv2.cvtColor(data.reshape(h, w), cv2.COLOR_GRAY2BGR)
    if ch == 3:
        arr = data.reshape(h, w, 3)
        return arr if enc.startswith("bgr") else cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    raise ValueError("unsupported image encoding: %s" % msg.encoding)


def bgr_to_image_msg(bgr, header=None):
    bgr = np.ascontiguousarray(bgr, dtype=np.uint8)
    h, w = bgr.shape[:2]
    ch = 1 if bgr.ndim == 2 else bgr.shape[2]
    msg = Image()
    msg.height = h
    msg.width = w
    msg.encoding = "mono8" if ch == 1 else "bgr8"
    msg.is_bigendian = 0
    msg.step = w * ch
    msg.data = bgr.tobytes()
    if header is not None:
        msg.header = header
    return msg


def apply_region_growing(img, seed_point, threshold=10, bbox=None):
    h, w = img.shape[:2]
    cx = max(0, min(int(seed_point[0]), w - 1))
    cy = max(0, min(int(seed_point[1]), h - 1))
    flood_img = img.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    cv2.floodFill(
        flood_img, mask, (cx, cy), 255,
        loDiff=(threshold, threshold, threshold),
        upDiff=(threshold, threshold, threshold),
        flags=flags,
    )
    res_mask = mask[1:-1, 1:-1].copy()
    if bbox is not None:
        roi_limit = np.zeros((h, w), np.uint8)
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, min(x1, w - 1)); y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1)); y2 = max(0, min(y2, h - 1))
        cv2.rectangle(roi_limit, (x1, y1), (x2, y2), 255, -1)
        res_mask = cv2.bitwise_and(res_mask, roi_limit)
    return res_mask


def get_region_color_info(img, contour):
    h, w = img.shape[:2]
    edge_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(edge_mask, [contour], -1, 255, 2)
    fill_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(fill_mask, [contour], -1, 255, -1)
    edge_pixels = img[edge_mask == 255]
    inner_pixels = img[fill_mask == 255]
    edge_rgb = (0, 0, 0); inner_rgb = (0, 0, 0)
    if len(edge_pixels) > 0:
        b = np.mean(edge_pixels, axis=0); edge_rgb = (int(b[2]), int(b[1]), int(b[0]))
    if len(inner_pixels) > 0:
        b = np.mean(inner_pixels, axis=0); inner_rgb = (int(b[2]), int(b[1]), int(b[0]))
    return edge_rgb, inner_rgb


class Forensic3DetectorV3Node(object):
    def __init__(self):
        self.model_path = rospy.get_param(
            "~model_path",
            "/home/zf/forensic_ws/models/forensic3_model_best.pth")
        self.arch = rospy.get_param("~arch", "resdcn_18")
        self.dataset = rospy.get_param("~dataset", "forensic_3_v3")
        self.num_classes = rospy.get_param("~num_classes", 3)
        self.vis_thresh = rospy.get_param("~vis_thresh", 0.3)
        self.use_rg = rospy.get_param("~use_region_growing", True)
        self.display = rospy.get_param("~display", True)
        image_topic = rospy.get_param("~image_topic", "/camera/color/image_raw")
        result_topic = rospy.get_param("~result_topic",
                                       "/forensic3_detect_v3/result_image")
        fps_topic = rospy.get_param("~fps_topic", "/forensic3_detect_v3/fps")

        if not os.path.isfile(self.model_path):
            rospy.logfatal("model file not found: %s", self.model_path)
            raise FileNotFoundError(self.model_path)

        self._build_detector()

        self.pub_result = rospy.Publisher(result_topic, Image, queue_size=1)
        self.pub_fps = rospy.Publisher(fps_topic, std_msgs.msg.Float32, queue_size=1)
        self.sub = rospy.Subscriber(image_topic, Image, self.cb,
                                    queue_size=1, buff_size=2 ** 24)
        self._last_t = time.time(); self._fps = 0.0
        rospy.loginfo("forensic3_detector_v3 subscribed to %s", image_topic)
        rospy.loginfo("publishing result to %s", result_topic)
        rospy.loginfo("model=%s arch=%s num_classes=%d vis_thresh=%.2f "
                      "region_growing=%s display=%s",
                      self.model_path, self.arch, self.num_classes,
                      self.vis_thresh, self.use_rg, self.display)

    def _build_detector(self):
        dummy_args = (
            "ctdet --arch {} --load_model {} --dataset {}".format(
                self.arch, self.model_path, self.dataset)).split()
        opt = opts().init(dummy_args)
        opt.num_classes = int(self.num_classes)
        opt.heads = {"hm": int(self.num_classes), "wh": 2, "reg": 2}
        opt.debug = 0
        opt.vis_thresh = float(self.vis_thresh)
        self.opt = opt
        rospy.loginfo("creating model (this takes a few seconds)...")
        self.detector = detector_factory["ctdet"](opt)
        rospy.loginfo("model loaded.")

    def cb(self, msg):
        try:
            img = image_msg_to_bgr(msg)
        except Exception as e:
            rospy.logerr("image conversion failed: %s", e); return
        if img is None or img.ndim != 3:
            return

        raw = self.detector.run(img)
        results = raw["results"] if isinstance(raw, dict) else {}
        vis = img.copy()

        n_detected = 0
        for cat_id, bboxes in results.items():
            if not isinstance(bboxes, (list, np.ndarray)):
                continue
            for obj in bboxes:
                if len(obj) < 5:
                    continue
                score = float(obj[4])
                if score < self.vis_thresh:
                    continue
                n_detected += 1
                x1, y1, x2, y2 = map(int, [obj[0], obj[1], obj[2], obj[3]])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cid = int(cat_id)
                label = CLASS_NAMES.get(cid, "obj%d" % cid)
                color = CLASS_COLORS.get(cid, (255, 255, 0))
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                txt = "{} {:.2f}".format(label, score)
                cv2.putText(vis, txt, (x1, max(y1 - 5, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)
                if self.use_rg:
                    self._draw_region_features(vis, img, (x1, y1, x2, y2),
                                               (cx, cy), label, score, color)

        now = time.time(); dt = now - self._last_t; self._last_t = now
        if dt > 0:
            inst = 1.0 / dt; self._fps = 0.8 * self._fps + 0.2 * inst
        self.pub_fps.publish(self._fps)
        cv2.putText(vis, "FPS %.1f  det %d" % (self._fps, n_detected),
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 1, cv2.LINE_AA)
        try:
            out = bgr_to_image_msg(vis, header=msg.header)
            self.pub_result.publish(out)
        except Exception as e:
            rospy.logerr("result publish failed: %s", e)
        if self.display:
            try:
                cv2.imshow("forensic3_detector_v3", vis); cv2.waitKey(1)
            except Exception as e:
                rospy.logwarn("display failed (no GUI?): %s", e)

    def _draw_region_features(self, vis, img, bbox, center, label, score, color):
        x1, y2_bot = bbox[0], min(bbox[3] + 20, vis.shape[0] - 4)
        try:
            mask = apply_region_growing(img.copy(), center, threshold=10, bbox=bbox)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                return
            px = int(M["m10"] / M["m00"]); py = int(M["m01"] / M["m00"])
            edge_rgb, inner_rgb = get_region_color_info(img, cnt)
            cv2.drawContours(vis, [cnt], -1, (0, 255, 0), 2)
            cv2.circle(vis, (px, py), 5, (0, 0, 255), -1)
            t1 = "Center:({},{}) Area:{}px".format(px, py, int(area))
            t2 = "EdgeRGB:({},{},{})".format(*edge_rgb)
            t3 = "InnerRGB:({},{},{})".format(*inner_rgb)
            base = y2_bot
            cv2.putText(vis, t1, (x1, base), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(vis, t2, (x1, base + 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(vis, t3, (x1, base + 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 1, cv2.LINE_AA)
        except Exception as e:
            rospy.logwarn("region growing failed: %s", e)


def main():
    rospy.init_node("forensic3_detector_v3", anonymous=False)
    rospy.loginfo("starting forensic3_detector_v3 node...")
    node = Forensic3DetectorV3Node()
    rospy.spin()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


if __name__ == "__main__":
    main()