#!/bin/bash
# run_forensic_auto.sh —— 自动物证采集联调UI启动 (forensic_ws)
# 与 run_forensic.sh 同结构, 仅 UI 脚本换成联调总代码 gluon_controller_forensic_auto.py
# 不影响原 gluon_controller_kk_copy.py (保留可随时切回手动版)
#
# 用法:
#   ./run_forensic_auto.sh          # 启动 roscore + 联调UI
#   ./run_forensic_auto.sh --core   # 强制重启 roscore
#
# UI 内点「自动识别」按钮: 启动机械臂+相机 -> 回零 -> 半圈扫描检测
#   -> 判类(微小/固体/液体) -> 粘取/夹取/涂抹 -> 回收位释放 -> 回零循环

HOME_DIR="${HOME:-/home/$USER}"
WS="$HOME_DIR/forensic_ws"
# ★ 2026-08-28: reBot 已替换 GLUON —— 自动物证采集 UI 切到 reBot 版
#   (含 reBot 手眼标定/乒乓扫描/v4检测画面; gluon 版仅存档)
SCRIPT="$WS/src/reBot/scripts/rebot_controller_forensic_auto.py"

source "$WS/scripts/env_setup.sh"
CONDA_PY="${FORENSIC_CONDA_PY:-}"
PYQT_LIB="${FORENSIC_PYQT_LIB:-}"
PYQTPLUGINS="${FORENSIC_PYQT_PLUGINS:-}"
set -u

RESTART_CORE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --core) RESTART_CORE=1; shift ;;
    *) echo "未知参数: $1" >&2; shift ;;
  esac
done

[[ -f "$SCRIPT" ]] || { echo "[错误] UI脚本不存在: $SCRIPT" >&2; exit 1; }
[[ -x "$CONDA_PY" ]] || { echo "[错误] conda python 不存在: $CONDA_PY (请改 $WS/scripts/env_setup.sh)" >&2; exit 1; }
[[ -f "$WS/devel/setup.bash" ]] || { echo "[错误] forensic_ws 未编译, 请先: cd $WS && catkin_make" >&2; exit 1; }

# ---------- source (forensic_ws + forensic3_detect_ws_v4, 与原版 kk_copy 一致) ----------
source /opt/ros/melodic/setup.bash
source "$WS/devel/setup.bash"
# 联调用 forensic3_auto_camera.launch (在 forensic3_detector_v4 包内), 需 source 该工作空间
FORENSIC3_WS="$HOME_DIR/forensic3_detect_ws_v4"
[[ -f "$FORENSIC3_WS/devel/setup.bash" ]] && source "$FORENSIC3_WS/devel/setup.bash" --extend

if ! rospack find gluon >/dev/null 2>&1 || ! rospack find orbbec_camera >/dev/null 2>&1; then
  echo "[错误] forensic_ws 包注册异常 (gluon/orbbec_camera 不可见)" >&2; exit 1
fi
if ! rospack find forensic3_detector_v4 >/dev/null 2>&1; then
  echo "[错误] forensic3_detector_v4 包不可见 (需 source ~/forensic3_detect_ws_v4/devel/setup.bash)" >&2; exit 1
fi

# ---------- roscore ----------
if [[ "$RESTART_CORE" == "1" ]]; then
  echo "[forensic-auto] 重启 roscore ..."
  pkill -9 -f 'roscore|rosmaster' 2>/dev/null; sleep 1
fi
if rostopic list >/dev/null 2>&1; then
  echo "[forensic-auto] roscore 已运行, 复用"
else
  echo "[forensic-auto] 启动 roscore ..."
  setsid nohup roscore >/tmp/forensic_roscore.log 2>&1 &
  for i in $(seq 1 20); do rostopic list >/dev/null 2>&1 && break; sleep 1; done
  rostopic list >/dev/null 2>&1 || { echo "[错误] roscore 启动失败" >&2; exit 1; }
fi

# ---------- Innfos ECU 网卡 (机械臂主关节通信) ----------
ECU_IP="192.168.1.30"
ECU_IFACE=$(ip -o link show | awk -F': ' '{print $2}' | grep -E '^enx[0-9a-f]{12}$' | head -1)
if [[ -n "$ECU_IFACE" ]]; then
  if ! ping -c1 -W2 "$ECU_IP" >/dev/null 2>&1; then
    echo "[forensic-auto] 配置 ECU 网卡 $ECU_IFACE ..."
    echo '1' | sudo -S bash -c "
      nmcli device set $ECU_IFACE managed no >/dev/null 2>&1
      ip addr flush dev $ECU_IFACE 2>/dev/null
      ip addr add 192.168.1.10/24 dev $ECU_IFACE 2>/dev/null
      ip link set $ECU_IFACE up 2>/dev/null
    " 2>/dev/null
  fi
  ping -c1 -W2 "$ECU_IP" >/dev/null 2>&1 && echo "[forensic-auto] ECU $ECU_IP 可达 ✓" \
    || echo "[forensic-auto] ECU 不可达, 请确认机械臂已通电/USB已连"
else
  echo "[forensic-auto] 未找到 USB-Ethernet 网卡 (enx*), 机械臂主关节可能无法启动"
fi

# ---------- 夹爪串口权限 (ttyUSB0) ----------
if [[ -e /dev/ttyUSB0 ]] && [[ ! -w /dev/ttyUSB0 ]]; then
  echo '1' | sudo -S chmod 666 /dev/ttyUSB0 2>/dev/null
  echo "[forensic-auto] /dev/ttyUSB0 已 chmod 666"
fi

# ---------- PyQt5 Qt 库路径 ----------
export LD_LIBRARY_PATH="$PYQT_LIB:$LD_LIBRARY_PATH"
export QT_QPA_PLATFORM_PLUGIN_PATH="$PYQTPLUGINS"
export DISPLAY="${DISPLAY:-:0}"

# ---------- 启动联调 UI ----------
echo "[forensic-auto] 启动联调UI (conda py3 + PyQt5 + torch) ..."
echo "  脚本: $SCRIPT"
echo "  停止: 关闭 UI 窗口 或 pkill -f gluon_controller_forensic_auto.py"
setsid nohup "$CONDA_PY" -u "$SCRIPT" >/tmp/forensic_auto_ui.log 2>&1 < /dev/null &
disown
PID=$!
sleep 12
if kill -0 "$PID" 2>/dev/null; then
  echo "[forensic-auto] 联调UI启动成功 ✓ (PID=$PID)"
else
  echo "[forensic-auto] 联调UI启动失败 ✗, 日志末尾:"
  tail -25 /tmp/forensic_auto_ui.log
fi

echo ""
echo "[forensic-auto] 在 UI 内点「自动识别」按钮即可启动全自动流程:"
echo "  启动机械臂+相机 -> 回零 -> 半圈扫描检测 -> 判类 -> 粘取/夹取/涂抹 -> 回收位释放 -> 回零循环"
echo "  切回手动版: pkill -f gluon_controller_forensic_auto.py && ~/forensic_ws/scripts/run_forensic.sh"
