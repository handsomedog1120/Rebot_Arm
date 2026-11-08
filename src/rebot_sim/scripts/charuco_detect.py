#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检测 ChArUco 标定板：输出每个 ArUco marker 的 ID 和边长（像素 + 毫米）。

用法:
    python3 charuco_detect.py <图片路径>              # 静态图片
    python3 charuco_detect.py                        # 打开摄像头实时检测, q 退出

毫米换算原理: 已知棋盘方格边长 SQUARE_LEN_MM(实测36.5mm), 用相邻 charuco
角点的像素距离求出 mm/px, 再乘以 marker 像素边长, 即得 marker 实际边长。
"""
import os
import sys
import numpy as np
import cv2 as cv

HAS_DISPLAY = bool(os.environ.get("DISPLAY"))

# ---------------- 与生成板子时保持一致 ----------------
SQUARES_X, SQUARES_Y = 5, 7
SQUARE_LEN_MM = 36.5        # 2026-09-27 卷尺实测, 与 aruco_calib_gemini.launch 一致
MARKER_LEN_MM = 23.0        # 2026-09-27 实测, 脚本会用像素测量做交叉验证
DICT_TYPE = cv.aruco.DICT_6X6_250

dictionary = cv.aruco.getPredefinedDictionary(DICT_TYPE)
board = cv.aruco.CharucoBoard_create(SQUARES_X, SQUARES_Y,
                                     SQUARE_LEN_MM / 1000.0,
                                     MARKER_LEN_MM / 1000.0, dictionary)
params = cv.aruco.DetectorParameters_create()


def edge_lengths_px(corners):
    """marker 四个角点 -> 四条边的像素长度"""
    c = corners.reshape(4, 2).astype(np.float64)
    return [np.linalg.norm(c[(i + 1) % 4] - c[i]) for i in range(4)]


def mm_per_pixel(gray):
    """用 charuco 角点间距估计 mm/px (同一行相邻角点距离 = 方格边长)"""
    corners, ids, _ = cv.aruco.detectMarkers(gray, dictionary, parameters=params)
    if not corners:
        return None, 0
    ok, ch_corners, ch_ids = cv.aruco.interpolateCornersCharuco(corners, ids, gray, board)
    if not ok or ch_corners is None or len(ch_corners) < 2:
        return None, len(corners)
    # charuco 角点按行排列: 同一行相邻角点 id 差 1, 像素距离 = 方格边长
    pts = ch_corners.reshape(-1, 2)
    cids = ch_ids.ravel()
    dists = []
    for i in range(len(cids) - 1):
        if cids[i + 1] - cids[i] == 1:          # 同行相邻
            dists.append(np.linalg.norm(pts[i + 1] - pts[i]))
    if not dists:
        return None, len(corners)
    return SQUARE_LEN_MM / float(np.median(dists)), len(corners)


def analyze(gray, color):
    corners, ids, rejected = cv.aruco.detectMarkers(gray, dictionary, parameters=params)
    if not corners:
        print("未检测到 marker")
        return color

    scale, _ = mm_per_pixel(gray)
    print(f"\n检测到 {len(corners)} 个 marker"
          + (f"  (mm/px = {scale:.4f})" if scale else "  (未能估计比例, 只输出像素)"))
    print(f"{'ID':>4} {'边长px(均值±差)':>20} {'边长mm(估)':>10}")

    for corner, mid in zip(corners, ids.ravel()):
        edges = edge_lengths_px(corner)
        mean_px = float(np.mean(edges))
        msg = f"{mid:>4} {mean_px:>12.1f} ±{np.ptp(edges):<5.1f}"
        if scale:
            mm = mean_px * scale
            msg += f" {mm:>8.1f}"
        print(msg)

    cv.aruco.drawDetectedMarkers(color, corners, ids)
    return color


def main():
    if len(sys.argv) > 1:
        img = cv.imread(sys.argv[1])
        if img is None:
            sys.exit(f"无法读取图片: {sys.argv[1]}")
        out = analyze(cv.cvtColor(img, cv.COLOR_BGR2GRAY), img)
        cv.imwrite("charuco_annotated.png", out)
        print("\n已保存标注图: charuco_annotated.png")
        cv.imshow("charuco", out)
        cv.waitKey(0)
    else:
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            sys.exit("无法打开摄像头, 请传入图片路径")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            analyze(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), frame)
            cv.imshow("charuco (q 退出, 每30帧刷新统计)", frame)
            if cv.waitKey(30) & 0xFF == ord('q'):
                break
        cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
