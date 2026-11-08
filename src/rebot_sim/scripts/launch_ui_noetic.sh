#!/bin/bash
# launch_ui_noetic.sh —— 在 noetic 环境下启动 reBot 联调 UI
# (与 launch_ui.sh 同逻辑, source 路径 melodic -> noetic)

# 清扫残留进程
pkill -9 -f rebot_arm_server 2>/dev/null
pkill -9 -f rebot_controller_forensic_auto 2>/dev/null
pkill -9 -f orbbec_camera_node 2>/dev/null
pkill -9 -f "forensic3_detect[.]launch" 2>/dev/null
sleep 1
rm -f "$HOME/rebot_arm_server.err"

# ---- can0 状态检查 (不阻塞) ----
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
# ★ 两个工作空间是 melodic 时代编译的, devel/setup.bash 在 noetic 下不更新
#   ROS_PACKAGE_PATH。直接设 src 路径让 roslaunch 子进程能找到包。
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
