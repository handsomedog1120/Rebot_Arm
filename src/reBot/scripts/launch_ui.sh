#!/bin/bash
# 启动 reBot 移植版 UI (centernet py3.8), 全 fd 重定向避免阻塞调用方
# ★ 自动恢复: can0 掉线(PCAN拔插/断电)自动拉起; roscore 未跑自动启动
pkill -9 -f rebot_arm_server 2>/dev/null
pkill -9 -f rebot_controller_forensic_auto 2>/dev/null
# ★ 清扫上次残留的相机进程: 孤儿 orbbec_camera_node 会占住USB设备,
#   导致新UI「启动相机」打不开(必须插拔USB的根因)
pkill -9 -f orbbec_camera_node 2>/dev/null
pkill -9 -f "forensic3_detect[.]launch" 2>/dev/null
sleep 1
rm -f "$HOME/rebot_arm_server.err"

# ---- can0 掉线自动恢复 (需要 sudo, 密码写死 1; 失败不影响UI启动) ----
# ★ 2026-08-31: PCAN-USB 适配器会在开机后随机冻结(dmesg: Rx urb aborted -32,
#   表现为启动机械臂时7电机全部超时无响应). operstate=up 是假象, 须USB级重置.
#   预防性: 每次启动UI都重置一次PCAN(约5s, 无害), 拿到全新RX管道;
#   reset_pcan.sh 顺带禁用全部USB autosuspend(集线器空闲挂起是冻结诱因).
echo 1 | sudo -S bash "$(dirname "$0")/reset_pcan.sh" >/dev/null 2>&1
CAN_STATE=$(cat /sys/class/net/can0/operstate 2>/dev/null || echo down)
echo "[launch] can0 最终状态: $CAN_STATE"

# ---- roscore 未跑自动启动 ----
if ! pgrep -x roscore >/dev/null 2>&1; then
    echo "[launch] roscore 未运行, 后台启动..."
    source /opt/ros/melodic/setup.bash 2>/dev/null
    setsid nohup roscore > /tmp/roscore.log 2>&1 </dev/null &
    sleep 5
fi

source "$HOME/forensic_ws/scripts/env_setup.sh"
source /opt/ros/melodic/setup.bash
export QT_QPA_PLATFORM_PLUGIN_PATH="$FORENSIC_PYQT_PLUGINS"
export QT_PLUGIN_PATH="${FORENSIC_PYQT_PLUGINS%/platforms}"
export LD_LIBRARY_PATH="$FORENSIC_PYQT_LIB:$LD_LIBRARY_PATH"
cd "$HOME/forensic_ws/src/reBot/scripts"
# ★ X11 认证: 后台启动必须带桌面会话的 XAUTHORITY, 否则 GUI 随机断连
#   ("The X11 connection broke" -> UI崩溃 -> arm_server跟着退出 -> 电机断电)
export DISPLAY=:0
export XAUTHORITY="${XAUTHORITY:-/run/user/1000/gdm/Xauthority}"
# ★ -u: 关闭python stdout块缓冲 —— nohup重定向到文件时print是全缓冲(8KB),
#   检测/定位诊断日志会迟迟不落盘(排查"定位失败无日志"时被坑过)
nohup "$FORENSIC_CONDA_PY" -u rebot_controller_forensic_auto.py \
    > "$HOME/rebot_ui.log" 2>&1 </dev/null &
echo "launched pid=$!"
