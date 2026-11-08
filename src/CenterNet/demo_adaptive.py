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
                        depth_img = load_depth_map_by_image_path(image_name)
                        mask = apply_adaptive_region_growing(
                            img_raw.copy(),
                            (cx, cy),
                            bbox=bbox,
                            depth_img=depth_img
                        )
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



def load_depth_map_by_image_path(image_name):
    """
    根据 RGB 图片路径尝试读取对应深度图。
    默认对应关系：
    images/val2024/xxx.jpg   -> depth/val2024/xxx.png
    images/train2024/xxx.jpg -> depth/train2024/xxx.png

    如果当前没有深度图，则返回 None。
    """
    base = os.path.splitext(os.path.basename(image_name))[0]

    candidates = []

    if '/images/' in image_name:
        depth_base = image_name.replace('/images/', '/depth/')
        depth_dir = os.path.dirname(depth_base)
        candidates.append(os.path.join(depth_dir, base + '.png'))
        candidates.append(os.path.join(depth_dir, base + '_depth.png'))

    for dp in candidates:
        if os.path.exists(dp):
            depth = cv2.imread(dp, cv2.IMREAD_UNCHANGED)
            if depth is not None:
                print("读取深度图:", dp)
                return depth

    return None


def normalize_depth(depth_img):
    """
    将深度图统一转换为米。
    如果深度图是 uint16，通常单位为 mm，这里除以 1000。
    """
    if depth_img is None:
        return None

    depth = depth_img.astype(np.float32)

    if np.nanmax(depth) > 100:
        depth = depth / 1000.0

    return depth


def apply_adaptive_region_growing(img, seed_point, bbox=None, depth_img=None,
                                  min_color_thr=10,
                                  max_color_thr=38,
                                  color_k=1.6,
                                  depth_ratio=0.04):
    """
    自适应区域生长算法。

    改进点：
    1. 根据检测框内颜色波动自动确定颜色生长阈值；
    2. 使用梯度边缘约束，减少区域外泄；
    3. 如果存在深度图，则加入深度一致性约束；
    4. 最后进行形态学闭运算和开运算，平滑分割结果。
    """
    h, w = img.shape[:2]

    cx, cy = int(seed_point[0]), int(seed_point[1])
    cx = max(0, min(cx, w - 1))
    cy = max(0, min(cy, h - 1))

    if bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
    else:
        x1, y1, x2, y2 = 0, 0, w - 1, h - 1

    if x2 <= x1 or y2 <= y1:
        return np.zeros((h, w), np.uint8)

    # 转 Lab 色彩空间，比 BGR 更适合衡量颜色差异
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

    # 灰度图用于计算边缘梯度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)

    roi_lab = lab[y1:y2 + 1, x1:x2 + 1]
    roi_grad = grad[y1:y2 + 1, x1:x2 + 1]

    # 自适应颜色阈值：由候选框内部颜色波动决定
    color_std = np.mean(np.std(roi_lab.reshape(-1, 3), axis=0))
    color_thr = float(np.clip(color_std * color_k, min_color_thr, max_color_thr))

    # 边缘阈值：梯度较大的地方认为可能是边界
    edge_thr = float(np.percentile(roi_grad, 75))

    # 种子点颜色取 5×5 邻域均值，降低单点噪声影响
    sx1, sy1 = max(0, cx - 2), max(0, cy - 2)
    sx2, sy2 = min(w - 1, cx + 2), min(h - 1, cy + 2)
    seed_lab = np.mean(lab[sy1:sy2 + 1, sx1:sx2 + 1].reshape(-1, 3), axis=0)

    # 深度约束
    depth = normalize_depth(depth_img)
    use_depth = depth is not None and depth.shape[:2] == (h, w)

    seed_depth = None
    depth_thr = None

    if use_depth:
        patch = depth[sy1:sy2 + 1, sx1:sx2 + 1]
        valid = patch[patch > 0]

        if valid.size > 0:
            seed_depth = float(np.median(valid))
            depth_thr = max(0.015, seed_depth * depth_ratio)
        else:
            use_depth = False

    mask = np.zeros((h, w), np.uint8)
    visited = np.zeros((h, w), np.uint8)

    q = [(cx, cy)]
    visited[cy, cx] = 1
    mask[cy, cx] = 255

    head = 0

    while head < len(q):
        px, py = q[head]
        head += 1

        neighbors = [
            (px + 1, py),
            (px - 1, py),
            (px, py + 1),
            (px, py - 1)
        ]

        for nx, ny in neighbors:
            if nx < x1 or nx > x2 or ny < y1 or ny > y2:
                continue

            if visited[ny, nx]:
                continue

            visited[ny, nx] = 1

            color_dist = np.linalg.norm(lab[ny, nx] - seed_lab)

            if color_dist > color_thr:
                continue

            if grad[ny, nx] > edge_thr:
                continue

            if use_depth:
                d = depth[ny, nx]
                if d <= 0:
                    continue
                if abs(float(d) - seed_depth) > depth_thr:
                    continue

            mask[ny, nx] = 255
            q.append((nx, ny))

    # 形态学处理，使区域更连续
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    print("自适应区域生长参数: color_thr={:.2f}, edge_thr={:.2f}, use_depth={}".format(
        color_thr, edge_thr, use_depth
    ))

    return mask


if __name__ == '__main__':
    opt = opts().parse()
    Dataset = get_dataset(opt.dataset, opt.task)
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    opt.debug = 0
    demo(opt)
