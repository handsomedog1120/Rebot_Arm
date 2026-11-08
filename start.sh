#!/bin/bash
# ★ 双击此文件即可启动刑侦物证采集系统
# (若双击无反应, 右键 → 属性 → 权限 → 勾选"允许作为程序执行文件")
# 终端方式: bash ~/forensic_ws/start.sh
~/forensic_ws/scripts/run_forensic_auto.sh

cd "$(dirname "$0")"
exec bash scripts/run_forensic.sh "$@"
