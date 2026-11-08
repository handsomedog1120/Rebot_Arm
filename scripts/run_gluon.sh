#!/bin/bash
# run_gluon.sh —— 启动 GLUON 机械臂控制 GUI (gluon_controller_kk_copy.py)
#
# 背景: 该脚本是 PyQt5 + ROS 节点 + 串口舵机 (scservo_sdk) 的 python3 程序.
#   - ROS Melodic 自带 rospy 是 python2, 不能直接跑.
#   - 需用 conda centernet_4060 的 py3 (rospy + PyQt5 + pyserial).
#   - conda 的 PyQt5 自带 Qt 库与 env/lib 里的冲突, 必须让 PyQt5 自带 Qt 优先加载
#     (否则报 "symbol Qt_5 not defined in libQt5Core" relocation error).
# 因此不走 roslaunch (它用 py2 调度), 改为手动起 roscore + 用 conda python 直接跑脚本.
#
# 用法:
#   ./run_gluon.sh          # 先起 roscore (若未运行), 再启动 GUI
#   ./run_gluon.sh --core   # 强制重启 roscore

set -u

HOME_DIR="$HOME"
CATKIN_WS="$HOME_DIR/forensic_ws"
SCRIPT="$CATKIN_WS/src/gluon/ros_gluon/gluon/scripts/gluon_controller_kk_copy.py"
source ~/forensic_ws/scripts/env_setup.sh
CONDA_PY="$FORENSIC_CONDA_PY"
PYQT_LIB="$FORENSIC_PYQT_LIB"
PYQTPLUGINS="$FORENSIC_PYQT_PLUGINS"

RESTART_CORE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --core) RESTART_CORE=1; shift ;;
    *) echo "未知参数: $1" >&2; shift ;;
  esac
done

# ---------- 环境检查 ----------
[[ -f "$SCRIPT" ]] || { echo "[错误] 找不到控制脚本: $SCRIPT" >&2; exit 1; }
[[ -x "$CONDA_PY" ]] || { echo "[错误] 找不到 conda python: $CONDA_PY" >&2; exit 1; }
[[ -f "$CATKIN_WS/devel/setup.bash" ]] || { echo "[错误] catkin_py 未构建, 请先: cd $CATKIN_WS && catkin_make" >&2; exit 1; }

# ---------- source ----------
source /opt/ros/melodic/setup.bash
source "$CATKIN_WS/devel/setup.bash"

if ! rospack find gluon >/dev/null 2>&1; then
  echo "[错误] 找不到 ROS 包 gluon, 请先构建: cd $CATKIN_WS && catkin_make" >&2; exit 1
fi

# ---------- Innfos ECU 通信网卡配置 (关键修复) ----------
# Innfos ECU(6 个主关节执行器网关)通过 USB-Ethernet 网卡直连, 默认 IP 192.168.1.30.
# 若不配 IP 或被 NM 的 DHCP 清掉, gluon_node 报 "Connected error code:803" 找不到 ECU.
# 策略: 优先用 NM 静态连接 innfos-ecu (持久); 若不存在则临时手配.
ECU_IP="192.168.1.30"
ECU_IFACE=$(ip -o link show | awk -F': ' '{print $2}' | grep -E '^enx[0-9a-f]{12}$' | head -1)
if [[ -n "$ECU_IFACE" ]]; then
  if ping -c 1 -W 2 "$ECU_IP" >/dev/null 2>&1; then
    echo "[gluon] ECU $ECU_IP 已可达 ✓"
  else
    SUDOPW="${GLUON_SUDO_PW:-6}"
    echo "$SUDOPW" | sudo -S bash -c "
      # 优先 up NM 静态连接 (持久), 没有则临时手配
      nmcli connection up innfos-ecu >/dev/null 2>&1 || {
        ip addr flush dev $ECU_IFACE 2>/dev/null
        ip addr add 192.168.1.10/24 dev $ECU_IFACE 2>/dev/null
        ip link set $ECU_IFACE up 2>/dev/null
      }
    " 2>/dev/null
    sleep 1
    if ping -c 1 -W 2 "$ECU_IP" >/dev/null 2>&1; then
      echo "[gluon] ECU $ECU_IP 已配通 ✓"
    else
      echo "[gluon] 警告: ECU $ECU_IP 不可达, 请确认机械臂已通电/连接" >&2
    fi
  fi
else
  echo "[gluon] 警告: 未找到 USB-Ethernet 网卡(enx*), gluon_node 可能报 0x803" >&2
fi

# ---------- 夹爪串口权限 (m1ko 应在 dialout 组) ----------
if [[ ! -w /dev/ttyUSB0 ]]; then
  echo "[gluon] /dev/ttyUSB0 无写权限, 尝试 chmod ..."
  echo "${GLUON_SUDO_PW:-6}" | sudo -S chmod 666 /dev/ttyUSB0 2>/dev/null
fi

# ---------- roscore ----------
if [[ "$RESTART_CORE" == "1" ]]; then
  echo "[gluon] 请求重启 roscore ..."
  pkill -9 -f 'roscore|rosmaster' 2>/dev/null; sleep 1
fi
if rostopic list >/dev/null 2>&1; then
  echo "[gluon] roscore 已在运行, 复用"
else
  echo "[gluon] 启动 roscore ..."
  setsid nohup roscore >/tmp/gluon_roscore.log 2>&1 &
  for i in $(seq 1 20); do rostopic list >/dev/null 2>&1 && break; sleep 1; done
  rostopic list >/dev/null 2>&1 || { echo "[错误] roscore 启动失败, 见 /tmp/gluon_roscore.log"; exit 1; }
fi

# ---------- 检测串口 (机械臂已连接?) ----------
echo "[gluon] 可用串口设备:" 
ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null | sed 's/^/    /' || echo "    (无, 请确认机械臂 USB 已连接)"

# ---------- 关键: PyQt5 自带 Qt 优先, 修复 xcb 插件冲突 ----------
export LD_LIBRARY_PATH="$PYQT_LIB:$LD_LIBRARY_PATH"
export QT_QPA_PLATFORM_PLUGIN_PATH="$PYQTPLUGINS"
export DISPLAY="${DISPLAY:-:0}"

echo "[gluon] 启动控制 GUI (conda py3 + PyQt5) ..."
echo "        脚本: $SCRIPT"
echo "  停止方式: 关闭 GUI 窗口, 或 pkill -f gluon_controller_kk_copy.py"
setsid nohup "$CONDA_PY" "$SCRIPT" >/tmp/gluon_controller.log 2>&1 < /dev/null &
disown
PID=$!
echo "  PID=$PID"

# 等待 GUI 初始化 (PyQt5 + tf2 + 曲轴插件加载需~8-10s)
sleep 10
if kill -0 "$PID" 2>/dev/null; then
  echo "[gluon] 启动成功 ✓  (日志: /tmp/gluon_controller.log)"
else
  echo "[gluon] 启动失败 ✗, 日志末尾:"
  tail -15 /tmp/gluon_controller.log
fi
echo "  停止: pkill -f 'gluon_controller_kk_copy.py'; 停 roscore: pkill -9 -f 'roscore|rosmaster'"