#!/bin/bash
# launch_ui_noetic.sh —— 在 noetic 环境下启动 reBot 联调 UI
# (与 launch_ui.sh 同逻辑, source 路径 melodic -> noetic)

# 清扫残留进程
# ★ 2026-10-16: 后端已切换为 rebot_arm_control.py, 需一并清扫
#   (残留进程会占住 CAN 总线, 新后端 connect 全电机无响应)
pkill -9 -f rebot_arm_server 2>/dev/null
pkill -9 -f rebot_arm_control 2>/dev/null
pkill -9 -f rebot_controller_forensic_auto 2>/dev/null
pkill -9 -f orbbec_camera_node 2>/dev/null
pkill -9 -f "forensic3_detect[.]launch" 2>/dev/null
sleep 1
rm -f "$HOME/rebot_arm_server.err"

# ---- can0 复位 (2026-10-26: 先显式 down 再 up, 确保干净拉起) ----
# ★ 注意: 沿用本脚本既有 sudo 密码约定 "1", 密码不同请改 echo 1
echo 1 | sudo -S ip link set can0 down 2>/dev/null
sleep 0.5
echo 1 | sudo -S ip link set can0 up type can bitrate 1000000 restart-ms 100 2>/dev/null
# USB 级重置(等同插拔, 解适配器冻结)仍走 reset_pcan.sh, 其内部会再 down/up 一遍
echo 1 | sudo -S bash "$(dirname "$0")/reset_pcan.sh" >/dev/null 2>&1
CAN_STATE=$(cat /sys/class/net/can0/operstate 2>/dev/null || echo down)
echo "[launch] can0 状态: $CAN_STATE"

# ---- roscore ----
if ! pgrep -x roscore >/dev/null 2>&1; then
    echo "[launch] roscore 未运行, 后台启动..."
    source /opt/ros/noetic/setup.bash
    setsid nohup roscore > /tmp/roscore.log 2>&1 </dev/null &
    sleep 5
else
    echo "[launch] roscore 已运行, 复用"
fi

# ---- 环境变量 ----
source /opt/ros/noetic/setup.bash
source "$HOME/forensic_ws/scripts/env_setup.sh"
export QT_QPA_PLATFORM_PLUGIN_PATH="$FORENSIC_PYQT_PLUGINS"
export QT_PLUGIN_PATH="${FORENSIC_PYQT_PLUGINS%/platforms}"
export LD_LIBRARY_PATH="$FORENSIC_PYQT_LIB:$LD_LIBRARY_PATH"
# ★ noetic rospy 在 /opt/ros/noetic/lib/python3/dist-packages, conda py3.9 需显式加入
export PYTHONPATH="/opt/ros/noetic/lib/python3/dist-packages:${PYTHONPATH:-}"
# ★ melodic 时代编译的工作空间: devel/setup.bash 能正确注册包和可执行文件路径,
#   但 noetic 下 ROS_PACKAGE_PATH 不更新 → 叠加 src 路径让新包也能发现.
#   2026-10-21: 必须 source devel/setup.bash, 否则 orbbec_camera_node 二进制
#   找不到 ("Cannot locate node of type [orbbec_camera_node]"), 且 roslaunch
#   子进程不能解析 $(find orbbec_camera)/launch/deeya.launch.
source "$HOME/forensic_ws/devel/setup.bash" 2>/dev/null
source "$HOME/forensic3_detect_ws_v4/devel/setup.bash" 2>/dev/null
export ROS_PACKAGE_PATH="$HOME/forensic_ws/src:$HOME/forensic3_detect_ws_v4/src:${ROS_PACKAGE_PATH:-}"
export DISPLAY=:0
export XAUTHORITY="${XAUTHORITY:-/run/user/1000/gdm/Xauthority}"

cd "$HOME/forensic_ws/src/reBot/scripts"

nohup "$FORENSIC_CONDA_PY" -u rebot_controller_forensic_auto.py \
    > /tmp/rebot_ui.log 2>&1 </dev/null &
echo "[launch] UI started, PID=$!"

sleep 8
if pgrep -f rebot_controller_forensic_auto >/dev/null 2>&1; then
    echo "[launch] UI 运行中 ✓"
else
    echo "[launch] UI 启动失败, 日志末尾:"
    tail -25 /tmp/rebot_ui.log
fi
