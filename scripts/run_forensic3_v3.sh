#!/bin/bash
# run_forensic3_v3.sh —— 一键启动 Orbbec 相机 + CenterNet 三物证实时检测 (v3)
# 物证 tie/liquid/hexagon, 模型 forensic_3_v3 (from scratch, ev1/ev2/ev3 数据).
# 与 v1/v2 仅话题命名空间 /forensic3_detect_v3/* 与模型路径不同, 可共存(但同时
# 只能启动一个 launch, 共用同一台相机). 启动 v3 前先停旧版:
#   pkill -9 -f 'roslaunch forensic3_detector |forensic3_detect_node'

set -e

HOME_DIR="$HOME"
ROS_WS_NEW="$HOME_DIR/forensic_ws"
F3_WS="$HOME_DIR/forensic_ws"
LAUNCH_PKG="forensic3_detector_v3"
LAUNCH_FILE="forensic3_detect.launch"
VIEWER="1"; DO_BUILD="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build)  DO_BUILD="1"; shift ;;
    --noview) VIEWER="0"; shift ;;
    *) echo "未知参数: $1" >&2; shift ;;
  esac
done

[[ -d "$ROS_WS_NEW/src/OrbbecSDK_ROS1" ]] || { echo "[错误] 找不到 $ROS_WS_NEW" >&2; exit 1; }
[[ -d "$F3_WS/src/forensic3_detector_v3" ]] || { echo "[错误] 找不到 $F3_WS/src" >&2; exit 1; }
[[ -e "$F3_WS/models/forensic3_model_best.pth" ]] || {
  echo "[错误] 模型不存在: $F3_WS/models/forensic3_model_best.pth" >&2
  echo "       请确认训练已完成且软链已建 (exp/ctdet/forensic_3_v3/model_best.pth)" >&2; exit 1; }

source /opt/ros/melodic/setup.bash
source "$ROS_WS_NEW/devel/setup.bash"

if [[ "$DO_BUILD" == "1" || ! -f "$F3_WS/devel/setup.bash" ]]; then
  echo "[run_forensic3_v3] 构建 $F3_WS ..."
  catkin_make -C "$F3_WS" --source "$F3_WS/src"
fi
source "$F3_WS/devel/setup.bash"

rospack find "$LAUNCH_PKG" >/dev/null 2>&1 || { echo "[错误] 找不到 ROS 包 $LAUNCH_PKG, 先: $0 --build" >&2; exit 1; }

if ! rostopic list >/dev/null 2>&1; then
  echo "[run_forensic3_v3] 启动 roscore ..."
  roscore >/dev/null 2>&1 &
  for i in $(seq 1 20); do rostopic list >/dev/null 2>&1 && break; sleep 1; done
  rostopic list >/dev/null 2>&1 || { echo "[错误] roscore 启动失败" >&2; exit 1; }
else
  echo "[run_forensic3_v3] roscore 已在运行"
fi

echo "[run_forensic3_v3] 启动 roslaunch $LAUNCH_PKG $LAUNCH_FILE (display:=false) ..."
roslaunch "$LAUNCH_PKG" "$LAUNCH_FILE" display:=false \
  >/tmp/run_forensic3_v3_launch.log 2>&1 &

echo "[run_forensic3_v3] === 自检开始 ==="
ok="1"
check() { local d="$1" c="$2"; printf "  %-28s " "$d"; eval "$c" >/dev/null 2>&1 && echo "OK" || { echo "FAIL"; ok="0"; }; }

for i in $(seq 1 60); do
  rostopic list 2>/dev/null | grep -q "/camera/color/image_raw" && break; sleep 1
done
check "相机 color 话题"  "rostopic list 2>/dev/null | grep -q '/camera/color/image_raw'"
check "检测节点进程"     "pgrep -f 'forensic3_detect_v3_node.py' >/dev/null"
check "结果图话题"       "rostopic list 2>/dev/null | grep -q '/forensic3_detect_v3/result_image'"
check "FPS 话题"         "rostopic list 2>/dev/null | grep -q '/forensic3_detect_v3/fps'"

GOT=""
for i in $(seq 1 40); do
  timeout 2 rostopic hz /forensic3_detect_v3/result_image 2>/dev/null | grep -q "average rate" && { GOT=1; break; }
  sleep 1
done
printf "  %-28s " "结果图帧率"
if [[ "$GOT" == "1" ]]; then
  timeout 4 rostopic hz /forensic3_detect_v3/result_image 2>/dev/null | grep "average rate" | sed 's/^/OK /'
else echo "FAIL (无数据)"; ok="0"; fi

printf "  %-28s " "检测 FPS"
FPS=$(timeout 4 rostopic echo -n 1 /forensic3_detect_v3/fps 2>/dev/null | grep data | head -1 | sed 's/data: /OK /')
echo "${FPS:-FAIL}"

echo "launch log: /tmp/run_forensic3_v3_launch.log"
[[ "$ok" == "1" ]] && echo "[run_forensic3_v3] 自检通过 ✓" || echo "[run_forensic3_v3] 自检有失败项 ✗, 见日志"

if [[ "$ok" == "1" && "$VIEWER" == "1" ]]; then
  echo "[run_forensic3_v3] 打开 image_view (/forensic3_detect_v3/result_image) ..."
  rosrun image_view image_view image:=/forensic3_detect_v3/result_image \
    >/tmp/run_forensic3_v3_viewer.log 2>&1 &
fi

echo "[run_forensic3_v3] 完成. 停止:"
echo "  pkill -9 -f 'roslaunch forensic3_detector_v3|forensic3_detect_v3_node|orbbec_camera_node|image_view|roscore|rosmaster'"
wait