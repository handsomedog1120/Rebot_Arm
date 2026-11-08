#!/bin/bash
# ★ PCAN-USB 适配器USB级重置(等同插拔) + 重新拉起 can0 + 全USB禁autosuspend
# 冻结症状: dmesg "Rx urb aborted (-32)"; can0 operstate=up 但收不到任何CAN回包.
# 诱因: PCAN与Orbbec相机共享USB2总线, 相机大流量饿死PCAN的USB传输;
#       集线器空闲autosuspend也会冻结Rx管道 —— 本脚本顺带全部禁掉.
# 用法: echo <sudo密码> | sudo -S bash reset_pcan.sh   (需root: authorized/ip link)
for d in /sys/bus/usb/devices/*/idVendor; do
  if [ "$(cat $d 2>/dev/null)" = "0c72" ]; then
    p=${d%idVendor}
    echo 0 > ${p}authorized 2>/dev/null
    sleep 1.5
    echo 1 > ${p}authorized 2>/dev/null
  fi
done
# 集线器/设备空闲挂起是适配器冻结诱因: 全部禁autosuspend (power/control=on)
for h in /sys/bus/usb/devices/*/power/control; do echo on > $h 2>/dev/null; done
sleep 2
ip link set can0 down 2>/dev/null
ip link set can0 up type can bitrate 1000000 restart-ms 100 2>/dev/null
exit 0
