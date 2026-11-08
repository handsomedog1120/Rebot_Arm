#!/bin/bash
# launch.sh —— desktop 启动入口 (展开 $HOME 后调 run_forensic.sh)
# desktop 文件不展开 $HOME, 但能执行绝对路径脚本, 所以用本脚本做一层桥接
exec "$HOME/forensic_ws/scripts/run_forensic.sh" "$@"