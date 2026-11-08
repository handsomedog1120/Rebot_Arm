#!/bin/bash
# run_forensic.sh —— 刑侦现场物证采集系统一键启动 (forensic_ws 工作空间)
# 统一启动: roscore → 手眼标定TF发布 → 机械臂控制GUI(conda py3)
# 由 UI 内部按钮触发: 启动机械臂 (display.launch, 含手眼标定TF) + 启动相机
# 可选: 实时物证检测 (命令行方式, 独立于UI)
#
# 用法:
#   ./run_forensic.sh             # 启动 roscore + UI (常用)
#   ./run_forensic.sh --core      # 强制重启 roscore
#   ./run_forensic.sh --detect    # 额外启动 v3 实时物证检测 (相机+检测, 开 image_view)
#
# 全部路径指向 ~/forensic_ws/, 与原 ros_ws_new/catkin_py/catkin_calib/forensic3_detect_ws_v3 完全分隔

HOME_DIR="${HOME:-/home/$USER}"
WS="$HOME_DIR/forensic_ws"          # ★ 新独立工作空间
SCRIPT="$WS/src/gluon/ros_gluon/gluon/scripts/gluon_controller_kk_copy.py"

# ★ 先 source 路径配置 (不设 -u 避免 env_setup 内未定义变量干扰)
source "$WS/scripts/env_setup.sh"
CONDA_PY="${FORENSIC_CONDA_PY:-}"
PYQT_LIB="${FORENSIC_PYQT_LIB:-}"
PYQTPLUGINS="${FORENSIC_PYQT_PLUGINS:-}"

set -u   # source 完后再设 -u

RESTART_CORE=0; RUN_DETECT=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --core)   RESTART_CORE=1; shift ;;
    --detect) RUN_DETECT=1; shift ;;
    *) echo "未知参数: $1" >&2; shift ;;
  esac
done

[[ -f "$SCRIPT" ]] || { echo "[错误] UI脚本不存在: $SCRIPT" >&2; exit 1; }
[[ -x "$CONDA_PY" ]] || { echo "[错误] conda python 不存在: $CONDA_PY (请改 $WS/scripts/env_setup.sh)" >&2; exit 1; }
[[ -f "$WS/devel/setup.bash" ]] || { echo "[错误] forensic_ws 未编译, 请先: cd $WS && catkin_make" >&2; exit 1; }

# ---------- source (仅 forensic_ws) ----------
source /opt/ros/melodic/setup.bash
source "$WS/devel/setup.bash"      # 已含 orbbec/gluon/aruco_ros/forensic3_detector_v3/easy_handeye/visp_* 的产物

if ! rospack find gluon >/dev/null 2>&1 || ! rospack find orbbec_camera >/dev/null 2>&1; then
  echo "[错误] forensic_ws 包注册异常 (gluon/orbbec_camera 不可见)" >&2; exit 1
fi

# ---------- roscore ----------
if [[ "$RESTART_CORE" == "1" ]]; then
  echo "[forensic] 重启 roscore ..."
  pkill -9 -f 'roscore|rosmaster' 2>/dev/null; sleep 1
fi
if rostopic list >/dev/null 2>&1; then
  echo "[forensic] roscore 已运行, 复用"
else
  echo "[forensic] 启动 roscore ..."
  setsid nohup roscore >/tmp/forensic_roscore.log 2>&1 &
  for i in $(seq 1 20); do rostopic list >/dev/null 2>&1 && break; sleep 1; done
  rostopic list >/dev/null 2>&1 || { echo "[错误] roscore 启动失败" >&2; exit 1; }
fi

# ---------- Innfos ECU 网卡 (机械臂主关节通信) ----------
ECU_IP="192.168.1.30"
ECU_IFACE=$(ip -o link show | awk -F': ' '{print $2}' | grep -E '^enx[0-9a-f]{12}$' | head -1)
if [[ -n "$ECU_IFACE" ]]; then
  if ! ping -c1 -W2 "$ECU_IP" >/dev/null 2>&1; then
    echo "[forensic] 配置 ECU 网卡 $ECU_IFACE ..."
    echo '6' | sudo -S bash -c "
      nmcli device set $ECU_IFACE managed no >/dev/null 2>&1
      ip addr flush dev $ECU_IFACE 2>/dev/null
      ip addr add 192.168.1.10/24 dev $ECU_IFACE 2>/dev/null
      ip link set $ECU_IFACE up 2>/dev/null
    " 2>/dev/null
  fi
  ping -c1 -W2 "$ECU_IP" >/dev/null 2>&1 && echo "[forensic] ECU $ECU_IP 可达 ✓" \
    || echo "[forensic] ECU 不可达, 请确认机械臂已通电/USB已连"
else
  echo "[forensic] 未找到 USB-Ethernet 网卡 (enx*), 机械臂主关节可能无法启动"
fi

# ---------- 夹爪串口权限 (ttyUSB0) ----------
if [[ -e /dev/ttyUSB0 ]] && [[ ! -w /dev/ttyUSB0 ]]; then
  echo '6' | sudo -S chmod 666 /dev/ttyUSB0 2>/dev/null
  echo "[forensic] /dev/ttyUSB0 已 chmod 666"
fi

# ---------- PyQt5 Qt 库路径 (避免 xcb 冲突) ----------
export LD_LIBRARY_PATH="$PYQT_LIB:$LD_LIBRARY_PATH"
export QT_QPA_PLATFORM_PLUGIN_PATH="$PYQTPLUGINS"
export DISPLAY="${DISPLAY:-:0}"

# ---------- 可选: 实时物证检测 (--detect 模式) ----------
if [[ "$RUN_DETECT" == "1" ]]; then
  echo "[forensic] 启动 v3 实时物证检测 (后台) ..."
  setsid bash -c "source $WS/devel/setup.bash && roslaunch forensic3_detector_v3 forensic3_detect.launch display:=false vis_thresh:=0.60 > /tmp/forensic_detect.log 2>&1" < /dev/null &
  disown
  sleep 5
  rosrun image_view image_view image:=/forensic3_detect_v3/result_image >/tmp/forensic_view.log 2>&1 &
  disown
  echo "[forensic] 实时检测已启动, 图像窗口在桌面"
fi

# ---------- 启动 UI ----------
echo "[forensic] 启动 UI (conda py3 + PyQt5) ..."
echo "  脚本: $SCRIPT"
echo "  停止: 关闭 UI 窗口 或 pkill -f gluon_controller_kk_copy.py"
setsid nohup "$CONDA_PY" "$SCRIPT" >/tmp/forensic_ui.log 2>&1 < /dev/null &
disown
PID=$!
sleep 10
if kill -0 "$PID" 2>/dev/null; then
  echo "[forensic] UI 启动成功 ✓ (PID=$PID)"
else
  echo "[forensic] UI 启动失败 ✗, 日志末尾:"
  tail -15 /tmp/forensic_ui.log
fi

echo "[forensic] 完成. 在 UI 内点「启动机械臂」激活执行器+手眼标定TF, 点「启动相机」打开 astra"
echo "  全部停止: pkill -f 'gluon_controller_kk_copy.py'; pkill -9 -f 'roscore|rosmaster'"
echo "  手眼标定结果: $WS/docs/handeye_result.txt"