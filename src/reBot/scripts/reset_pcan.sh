#!/bin/bash
# ★ CAN-USB 适配器USB级重置(等同插拔) + 重新拉起 can0 + 全USB禁autosuspend
# 冻结症状: dmesg "Rx urb aborted (-32)" / "usb xmit fail"; can0 operstate=up
#           但收不到任何CAN回包(或can0直接消失).
# 诱因: CAN适配器与Orbbec相机共享USB2总线, 相机大流量饿死适配器的USB传输;
#       集线器空闲autosuspend也会冻结Rx管道 —— 本脚本顺带全部禁掉.
# 用法: echo <sudo密码> | sudo -S bash reset_pcan.sh   (需root: authorized/ip link)
#
# ★ 2026-11-05 修正:
#   1) 原脚本只匹配 VID=0c72(旧 PEAK PCAN), 对当前 candleLight(gs_usb,
#      1d50:606f)完全无效却仍 exit 0 → UI 误判"重置成功"后重试必然再失败.
#   2) 未找到适配器时明确 exit 1 并提示检查线材/USB口.
#   3) can0 重新枚举有窗口期: 轮询等待出现, 不再立刻 ip link 落空.
#   4) can0 up 与 operstate 校验失败均 exit 1, UI 侧 _reset_pcan_usb() 依据
#      返回码决定是否重试连接.
set -u

log() { echo "[reset_pcan] $*" >&2; }

found=0
for d in /sys/bus/usb/devices/*/idVendor; do
  [ -r "$d" ] || continue
  vid=$(cat "$d" 2>/dev/null)
  pid=$(cat "${d%idVendor}idProduct" 2>/dev/null)
  # 0c72:* = PEAK PCAN-USB; 1d50:606f = candleLight (gs_usb 驱动)
  if [ "$vid" = "0c72" ] || { [ "$vid" = "1d50" ] && [ "$pid" = "606f" ]; }; then
    p=${d%idVendor}
    log "找到 CAN 适配器 ${vid}:${pid} @ ${p%%authorized}"
    echo 0 > "${p}authorized" 2>/dev/null
    sleep 1.5
    echo 1 > "${p}authorized" 2>/dev/null
    found=1
  fi
done

if [ "$found" -eq 0 ]; then
  log "未找到 CAN 适配器(0c72:* 或 1d50:606f) — 设备未枚举, 请检查USB线/换机身独立USB口后重插"
  exit 1
fi

# 集线器/设备空闲挂起是适配器冻结诱因: 全部禁autosuspend (power/control=on)
for h in /sys/bus/usb/devices/*/power/control; do echo on > "$h" 2>/dev/null; done

# 重新枚举后 can0 需要时间重建, 最多等 10s
can_ok=0
for _ in $(seq 1 20); do
  if ip link show can0 >/dev/null 2>&1; then can_ok=1; break; fi
  sleep 0.5
done
if [ "$can_ok" -ne 1 ]; then
  log "can0 未重新出现(适配器未重建为 SocketCAN 接口) — 请拔插适配器"
  exit 1
fi

ip link set can0 down 2>/dev/null
if ! ip link set can0 up type can bitrate 1000000 restart-ms 100 2>/dev/null; then
  log "can0 up 失败(bitrate=1000000)"
  exit 1
fi

# operstate 校验: 让上层明确知道链路是否真的可用
sleep 0.5
state=$(cat /sys/class/net/can0/operstate 2>/dev/null)
if [ "$state" != "up" ]; then
  log "can0 operstate=${state:-未知}(期望 up) — USB 已重建但 CAN 链路未就绪"
  exit 1
fi

log "完成: can0 已就绪(bitrate=1000000, operstate=up)"
exit 0
