#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fk_helper.py —— 关节角 -> 末端位姿 的独立 FK 服务(rebot_pin 环境运行)

★ 存在理由
  reBot 的 FK 依赖 pinocchio, 只装在 rebot_pin 环境; 而采样脚本要跑在
  有 rospy/cv2 的 centernet_4060 环境. 两个环境互斥, 所以把 FK 单独拆成
  这个子进程 —— 父进程通过 JSON-line 传关节角, 这里回末端位姿.

★ 为什么不直接用 rebot_arm_control.py
  那个后端会连 CAN 总线. UI 运行时已经占着总线, 第二个实例必然
  "not connected". 本脚本【只做纯数学 FK】, 完全不碰硬件, 可以任意多开.

协议(stdin 每行一个 JSON, stdout 每行一个 JSON):
  {"cmd":"fk","q":[j1..j6]}  -> {"ok":true,"pos":[x,y,z],"rpy":[r,p,y]}
  {"cmd":"info"}             -> {"ok":true,"nq":..,"frame":".."}
  {"cmd":"quit"}             -> 退出
"""
import sys
import json
import os

import numpy as np
import pinocchio as pin

_PKG = os.path.expanduser('~/forensic_ws/src/reBot/reBotArm_control_py')
_URDF = os.path.join(_PKG, 'urdf', 'RS', 'urdf', 'ReBot_Arm_RS.urdf')
_END_FRAME = 'gripper_end'


def main():
    if not os.path.exists(_URDF):
        print(json.dumps({"ok": False, "err": "URDF not found: %s" % _URDF}),
              flush=True)
        return 1
    try:
        model = pin.buildModelFromUrdf(_URDF)
    except Exception as e:
        print(json.dumps({"ok": False, "err": "buildModel failed: %s" % e}),
              flush=True)
        return 1

    fid = model.getFrameId(_END_FRAME)
    if fid >= model.nframes:
        fid = model.nframes - 1
    # 就绪信号(父进程据此判断可用)
    print(json.dumps({"ok": True, "ready": True, "nq": int(model.nq),
                      "frame": model.frames[fid].name}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line.startswith('{'):
            continue
        try:
            req = json.loads(line)
        except Exception:
            continue
        cmd = req.get('cmd')
        if cmd == 'quit':
            break
        if cmd == 'info':
            print(json.dumps({"ok": True, "nq": int(model.nq),
                              "frame": model.frames[fid].name}), flush=True)
            continue
        if cmd != 'fk':
            print(json.dumps({"ok": False, "err": "unknown cmd"}), flush=True)
            continue
        try:
            q = np.asarray(req.get('q', []), dtype=np.float64).ravel()
            if q.shape[0] != model.nq:
                if q.shape[0] < model.nq:
                    q = np.concatenate([q, np.zeros(model.nq - q.shape[0])])
                else:
                    q = q[:model.nq]
            data = model.createData()
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            M = data.oMf[fid]
            pos = np.asarray(M.translation, dtype=np.float64).ravel()
            R = np.asarray(M.rotation, dtype=np.float64)
            rpy = np.asarray(pin.rpy.matrixToRpy(R), dtype=np.float64).ravel()
            print(json.dumps({"ok": True,
                              "pos": [float(v) for v in pos],
                              "rpy": [float(v) for v in rpy]}), flush=True)
        except Exception as e:
            print(json.dumps({"ok": False, "err": "fk failed: %s" % e}), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
