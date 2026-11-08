#!/bin/bash
# env_setup.sh —— forensic_ws 环境配置 (可移植)
# ★ 在新电脑上使用前, 修改本文件里的 CONDA_PY/PYQT_LIB/PYQTPLUGINS 路径即可
#   (指向新电脑上的 conda python + PyQt5 库路径)
#
# Source 方式: source ~/forensic_ws/scripts/env_setup.sh

# conda python 解释器 (含 rospy + PyQt5 + pyserial + ultralytics + torch.cuda)
export FORENSIC_CONDA_PY="${FORENSIC_CONDA_PY:-$HOME/anaconda3/envs/yolov8-gpu/bin/python}"

# PyQt5 自带 Qt 库路径 (解决 xcb 平台插件冲突)
export FORENSIC_PYQT_LIB="${FORENSIC_PYQT_LIB:-$HOME/anaconda3/envs/yolov8-gpu/lib/python3.10/site-packages/PyQt5/Qt5/lib}"
export FORENSIC_PYQT_PLUGINS="${FORENSIC_PYQT_PLUGINS:-$HOME/anaconda3/envs/yolov8-gpu/lib/python3.10/site-packages/PyQt5/Qt5/plugins/platforms}"

# ★ cv2 qt/plugins 冲突防御 (2026-10-21):
#   opencv-python/opencv-python-headless 都会自带一套 Qt plugins 目录,
#   其中的 libqxcb.so 在 headless 构建里是残缺/链接坏的 (28KB vs PyQt5
#   正常版 15.6KB), Qt 找到它就会崩 "Could not load xcb platform plugin".
#   即使是 opencv-python GUI 构建, 自带的插件也会覆盖 PyQt5 的.
#   每次 source 时自动删除, 保证 PyQt5 插件优先.
_CV2_QT_PLUGINS_DIR="$(dirname "$FORENSIC_CONDA_PY")/../lib/python3.10/site-packages/cv2/qt/plugins"
if [ -d "$_CV2_QT_PLUGINS_DIR" ]; then
    rm -rf "$_CV2_QT_PLUGINS_DIR" 2>/dev/null
fi
unset _CV2_QT_PLUGINS_DIR
