#!/bin/bash
# aruco_always_on.sh —— 标定管线常驻后台 (astra 相机仅彩色 + ArUco 检测)
# 独立于 UI 运行, 避免 UI「标定模式」按钮启停带来的卡顿/冲突。
# 启动: setsid nohup bash aruco_always_on.sh > /tmp/aruco_calib.log 2>&1 &
# 停止: pkill -f "aruco_calib.launc[h]"
source /opt/ros/melodic/setup.bash
source ~/forensic_ws/devel/setup.bash --extend
# aruco_single_cv2.py 靠 shebang "env python3" 取解释器, 需命中带 cv2.aruco 的 conda py3
export PATH='/home/zf/下载/enter/envs/centernet_4060/bin':$PATH
exec roslaunch aruco_ros aruco_calib.launch
