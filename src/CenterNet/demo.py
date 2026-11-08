# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import cv2
import numpy as np

from opts import opts
from detectors.detector_factory import detector_factory
from datasets.dataset_factory import get_dataset

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']

def apply_region_growing(img, seed_point, threshold=5, bbox=None):
    h, w = img.shape[:2]
    cx, cy = int(seed_point[0]), int(seed_point[1])
    cx = max(0, min(cx, w - 1))
    cy = max(0, min(cy, h - 1))

    flood_img = img.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)

    cv2.floodFill(
        flood_img, mask, (cx, cy), 255,
        loDiff=(threshold, threshold, threshold),
        upDiff=(threshold, threshold, threshold),
        flags=flags
    )

    res_mask = mask[1:-1, 1:-1].copy()

    if bbox is not None:
        roi_limit = np.zeros((h, w), np.uint8)
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        cv2.rectangle(roi_limit, (x1, y1), (x2, y2), 255, -1)
        res_mask = cv2.bitwise_and(res_mask, roi_limit)

    return res_mask
def get_region_color_info(img, contour):
    h, w = img.shape[:2]

    # 边缘mask
    edge_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(edge_mask, [contour], -1, 255, 2)

    # 内部mask
    fill_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(fill_mask, [contour], -1, 255, -1)

    # OpenCV读图默认是BGR
    edge_pixels = img[edge_mask == 255]
    inner_pixels = img[fill_mask == 255]

    edge_rgb = (0, 0, 0)
    inner_rgb = (0, 0, 0)

    if len(edge_pixels) > 0:
        edge_bgr_mean = np.mean(edge_pixels, axis=0)
        edge_rgb = (
            int(edge_bgr_mean[2]),
            int(edge_bgr_mean[1]),
            int(edge_bgr_mean[0])
        )

    if len(inner_pixels) > 0:
        inner_bgr_mean = np.mean(inner_pixels, axis=0)
        inner_rgb = (
            int(inner_bgr_mean[2]),
            int(inner_bgr_mean[1]),
            int(inner_bgr_mean[0])
        )

    return edge_rgb, inner_rgb


def get_class_map(dataset):
    if dataset == 'forensic_box':
        return {1: 'fixed_box'}
    elif dataset == 'forensic_micro':
        return {1: 'micro_paper'}
    elif dataset == 'forensic_liquid':
        return {1: 'liquid_drawing'}
    elif dataset == 'forensic':
        return {
            1: 'fixed_box',
            2: 'fixed_top',
            3: 'micro_paper',
            4: 'liquid_drawing'
        }
    else:
        return {}

def demo(opt):
    print(">>> 正在启动检测器...")
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = 0

    Detector = detector_factory[opt.task]
    detector = Detector(opt)
    class_map = get_class_map(opt.dataset)

    if opt.demo == 'webcam' or opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
        cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
        while True:
            _, img = cam.read()
            if img is None:
                break
            ret = detector.run(img)
            cv2.imshow('input', img)
            if cv2.waitKey(1) == 27:
                return
    else:
        if os.path.isdir(opt.demo):
            image_names = []
            for file_name in sorted(os.listdir(opt.demo)):
                ext = file_name[file_name.rfind('.') + 1:].lower()
                if ext in image_ext:
                    image_names.append(os.path.join(opt.demo, file_name))
        else:
            image_names = [opt.demo]

        output_dir = '../exp/ctdet/default/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for image_name in image_names:
            print("正在处理:", image_name)
            ret = detector.run(image_name)

            img_raw = cv2.imread(image_name)
            if img_raw is None:
                print("图像读取失败:", image_name)
                continue

            img_centernet = img_raw.copy()
            img_region = img_raw.copy()
            img_fusion = img_raw.copy()

            results = ret['results']

            for cat_id in results:
                for obj in results[cat_id]:
                    score = float(obj[4])
                    if score <= opt.vis_thresh:
                        continue

                    bbox = obj[:4]
                    x1, y1, x2, y2 = map(int, bbox)
                    cx, cy = int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)

                    label = class_map.get(cat_id, 'obj')
                    txt_label = "{} {:.2f}".format(label, score)

                    # 1) 只画 CenterNet 框
                    cv2.rectangle(img_centernet, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cv2.putText(img_centernet, txt_label, (x1, max(y1 - 5, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                    # 2) 融合图里也先画框
                    cv2.rectangle(img_fusion, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cv2.putText(img_fusion, txt_label, (x1, max(y1 - 5, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                    
                    # 3) 区域生长
                    try:
                        mask = apply_region_growing(img_raw.copy(), (cx, cy), threshold=10, bbox=bbox)
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        if contours:
                            cnt = max(contours, key=cv2.contourArea)
                            area = cv2.contourArea(cnt)
                            M = cv2.moments(cnt)

                            if M["m00"] != 0:
                                px = int(M["m10"] / M["m00"])
                                py = int(M["m01"] / M["m00"])

                                edge_rgb, inner_rgb = get_region_color_info(img_raw, cnt)

                                txt_info1 = "Center:({},{}) Area:{}px".format(px, py, int(area))
                                txt_info2 = "EdgeRGB:({},{},{})".format(edge_rgb[0], edge_rgb[1], edge_rgb[2])
                                txt_info3 = "InnerRGB:({},{},{})".format(inner_rgb[0], inner_rgb[1], inner_rgb[2])

                                # 只画区域生长结果
                                cv2.drawContours(img_region, [cnt], -1, (0, 255, 0), 2)
                                cv2.circle(img_region, (px, py), 5, (0, 0, 255), -1)
                                cv2.putText(img_region, txt_info1, (x1, min(y2 + 20, img_region.shape[0] - 30)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                                cv2.putText(img_region, txt_info2, (x1, min(y2 + 40, img_region.shape[0] - 15)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
                                cv2.putText(img_region, txt_info3, (x1, min(y2 + 60, img_region.shape[0] - 2)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

                                # 融合图也画区域生长结果
                                cv2.drawContours(img_fusion, [cnt], -1, (0, 255, 0), 2)
                                cv2.circle(img_fusion, (px, py), 5, (0, 0, 255), -1)
                                cv2.putText(img_fusion, txt_info1, (x1, min(y2 + 20, img_fusion.shape[0] - 30)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                                cv2.putText(img_fusion, txt_info2, (x1, min(y2 + 40, img_fusion.shape[0] - 15)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
                                cv2.putText(img_fusion, txt_info3, (x1, min(y2 + 60, img_fusion.shape[0] - 2)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

                                print("--- [识别结果] ---")
                                print("类别: {}, 分数: {:.3f}".format(label, score))
                                print("质心: ({}, {}), 面积: {} px".format(px, py, int(area)))
                                print("边缘RGB: {}, 内部RGB: {}".format(edge_rgb, inner_rgb))
                    except Exception as e:
                        print("区域生长绘图出错:", e)

            base_name = os.path.splitext(os.path.basename(image_name))[0]

            save_centernet = os.path.join(output_dir, base_name + "_centernet.jpg")
            save_region = os.path.join(output_dir, base_name + "_region.jpg")
            save_fusion = os.path.join(output_dir, base_name + "_fusion.jpg")

            cv2.imwrite(save_centernet, img_centernet)
            cv2.imwrite(save_region, img_region)
            cv2.imwrite(save_fusion, img_fusion)

            print(">>> CenterNet结果已保存至:", save_centernet)
            print(">>> 区域生长结果已保存至:", save_region)
            print(">>> 融合结果已保存至:", save_fusion)

if __name__ == '__main__':
    opt = opts().parse()
    Dataset = get_dataset(opt.dataset, opt.task)
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    opt.debug = 0
    demo(opt)
