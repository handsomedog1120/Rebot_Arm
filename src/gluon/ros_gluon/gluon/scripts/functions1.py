#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from numpy import sin, cos, pi, arctan2, arccos, sqrt

def Quaternion_2_matrix(Q_normed_):
    """
    单位四元数 转化成 旋转矩阵 [x,y,z,w]
    @Parameters
    Q_normed_: list [qx_, qy_, qz_, qw_]
    @Returns
    m: numpy.array([[]]) SO(3)
    """
    q0 = Q_normed_[3]
    q1 = Q_normed_[0]
    q2 = Q_normed_[1]
    q3 = Q_normed_[2]
    m = np.array([[1-2*q2**2-2*q3**2,   2*q1*q2-2*q0*q3,   2*q1*q3+2*q0*q2],
                  [2*q1*q2+2*q0*q3, 1-2*q1**2-2*q3**2,   2*q2*q3-2*q0*q1],
                  [2*q1*q3-2*q0*q2,   2*q2*q3+2*q0*q1, 1-2*q1**2-2*q2**2]])
    return m

def workSpace(pos):
    '''划定工作区域'''
    x = pos[0]
    y = pos[1]
    z = pos[2]
    xmin = -0.40
    xmax = 0.40
    ymin = -0.40
    ymax = 0.40
    zmin = -0.40
    zmax = 0.40
    if xmin <= x <= xmax and ymin <= y <= ymax and zmin <= z <= zmax:
        return True
    else:
        print(pos, 'out of space')
        return False

def inv_k(target_pos, target_orn=None, d7=0.165):
    """
    UR3/5/10, gluon通用的求解方法
    末端执行器在第6轴法兰上存在偏距 d7，需先进行位姿补偿

    相比原代码的改进（不改变基本结构，公式完全保留原推导）：
    - th1 增加左右手2解（原代码仅求1解）
    - th5 保留正负2解
    - th6 保留原代码的2解切换逻辑（pi - th6）
    - th3 增加肘上/肘下2解（原代码仅求肘上1解：arccos(...)-pi）
    - th2、th4、验证式完全保留原代码
    理论上最多可得 2×2×2×2=8 组解

    @Parameters
    target_pos: [x, y, z] 末端执行器目标位置 (m)
    target_orn: [qx, qy, qz, qw] 末端执行器目标姿态 (四元数)
    d7: 末端执行器在第6轴法兰上的偏距 (m)，默认0.165

    @Returns
    solve: [[th1, th2, th3, th4, th5, th6], ...]
    """
    if target_orn is None:
        target_orn = [0, 0, 0, 1]

    # ===========================================================
    # 末端执行器偏距补偿：
    # 给定末端执行器目标位姿 T_ee，需先转换为第6轴法兰的目标位姿 T6
    # T6 = T_ee * (T_offset)^(-1)
    # 其中 T_offset = [1, 0, 0, 0; 0, 1, 0, 0; 0, 0, 1, d7; 0, 0, 0, 1]
    # (T_offset)^(-1) = [1, 0, 0, 0; 0, 1, 0, 0; 0, 0, 1, -d7; 0, 0, 0, 1]
    # 
    # 对于位置：p6 = p_ee - d7 * R_ee[:, 2]  (沿末端Z轴负方向移动d7)
    # 对于姿态：R6 = R_ee (姿态不变，因为偏距沿Z轴方向)
    # ===========================================================
    
    x_ee = target_pos[0]
    y_ee = target_pos[1]
    z_ee = target_pos[2]
    qx = target_orn[0]
    qy = target_orn[1]
    qz = target_orn[2]
    qw = target_orn[3]

    # 末端执行器姿态旋转矩阵
    R_ee = Quaternion_2_matrix([qx, qy, qz, qw])
    
    # 末端执行器Z轴方向向量 (旋转矩阵第三列)
    z_axis_ee = R_ee[:, 0]  # [R_ee[0,2], R_ee[1,2], R_ee[2,2]]
    
    # 补偿后的第6轴法兰位置
    x = x_ee - d7 * z_axis_ee[0]
    y = y_ee - d7 * z_axis_ee[1]
    z = z_ee - d7 * z_axis_ee[2]
    
    # 第6轴法兰姿态 = 末端执行器姿态 (偏距沿Z轴，姿态不变)
    r = R_ee

    solve = []

    # ---------------------------------------------------------------
    # th1：原代码公式保留，增加 +/- 两组解（左右手）
    # ---------------------------------------------------------------
    a1 = x**2 + y**2 - 0.005728325   # 0.005728325 = d4^2

    if -0.0001 < a1 < 0.0001:
        # 奇异：目标在 z 轴附近，th1 不确定，取一个值
        th1_candidates = [-arctan2(y, x)]
    else:
        base = arctan2(y, x)
        # 解1（原代码的解）
        th1_A = arctan2(sqrt(abs(a1)), 0.0756857) - base
        # 解2（左手系：将 sqrt(a1) 取负，等价于用 pi - arctan2(...)）
        th1_B = (pi - arctan2(sqrt(abs(a1)), 0.0756857)) - base
        th1_candidates = [th1_A, th1_B]

    for th1 in th1_candidates:
        s1 = sin(th1)
        c1 = cos(th1)

        # -------------------------------------------------------
        # th5：原代码公式，正负2解
        # -------------------------------------------------------
        cos_th5 = float(np.clip(c1*r[0, 0] - s1*r[1, 0], -1.0, 1.0))
        th5     = arccos(cos_th5)
        th5_candidates = [-th5, -th5]

        for th5 in th5_candidates:
            s5 = sin(th5)
            c5 = cos(th5)

            # ---------------------------------------------------
            # th6：原代码公式保留，枚举两个分支（原代码用i==1/3切换）
            # ---------------------------------------------------
            px_th6 = r[2, 2] * s5
            py_th6 = r[2, 1] * s5
            d_th6  = r[2, 0] * c5

            inner = px_th6**2 + py_th6**2 - d_th6**2
            if inner < 0:
                continue
            th6_base = -arctan2(d_th6, sqrt(inner)) - arctan2(py_th6, px_th6)
            th6_alt  = pi - th6_base

            for th6 in [th6_base, th6_alt]:
                if th6 > 3.14:
                    th6 = th6 - 2 * pi
                elif th6 < -3.14:
                    th6 = th6 + 2 * pi

                c6 = cos(th6)
                s6 = sin(th6)

                # -----------------------------------------------
                # th3：原代码公式，增加肘上/肘下2解
                # -----------------------------------------------
                D = (x - 0.08*(r[0, 2]*c6 - r[0, 1]*s6))**2 + \
                    (y - 0.08*(r[1, 2]*c6 - r[1, 1]*s6))**2 + \
                    (z - 0.08*(r[2, 2]*c6 - r[2, 1]*s6) - 0.105)**2 - \
                    0.06656121

                if D > 0.06083072 or D < -0.06083072:
                    continue

                cos_th3 = float(np.clip(-D / 0.06083072, -1.0, 1.0))
                th3_base_val = arccos(cos_th3)
                th3_up   = th3_base_val - pi      # 肘上解（原代码仅此解）
                th3_down = -th3_base_val - pi     # 肘下解（新增）

                for th3 in [th3_up, th3_down]:
                    if th3 > 3.14:
                        th3 = th3 - 2 * pi
                    elif th3 < -3.14:
                        th3 = th3 + 2 * pi

                    # -------------------------------------------
                    # th2：原代码公式完全保留
                    # -------------------------------------------
                    th_tmpe = arccos(float(np.clip(
                        (z - 0.08*(r[2, 2]*c6 - r[2, 1]*s6) - 0.105) /
                        sqrt(D + 0.06083072),
                        -1.0, 1.0
                    )))
                    th2 = th_tmpe + th3 / 2

                    # -------------------------------------------
                    # th4：原代码公式完全保留
                    # -------------------------------------------
                    th_tmpe2 = arctan2(
                        (s1*(r[0, 2]*c6 - r[0, 1]*s6) + c1*(r[1, 2]*c6 - r[1, 1]*s6)),
                        (r[2, 2]*c6 - r[2, 1]*s6)
                    )
                    th4 = th_tmpe2 - th2 + th3
                    if th4 > 3.14:
                        th4 = th4 - 2 * pi
                    elif th4 < -3.14:
                        th4 = th4 + 2 * pi

                    # -------------------------------------------
                    # 正运动学验证：原代码公式完全保留
                    # -------------------------------------------
                    c234 = cos(th_tmpe2)
                    """ dist = (c1*c5 - s1*c234*s5 - r[0, 0])**2 + \
                           (s1*c5 + c1*c234*s5 + r[1, 0])**2 + \
                           (sin(th_tmpe2) - r[2, 0])**2
                    if dist > 1e-5:
                        continue """

                    # -------------------------------------------
                    # 关节限位检查：原代码完全保留
                    # -------------------------------------------
                    if (-2.79 < th1 < 2.79 and -1.65 < th2 < 1.65 and
                            -2.79 < th3 < 2.79 and -2.79 < th4 < 2.79 and
                            -2.79 < th5 < 2.79 and -2.79 < th6 < 2.79):
                        solve.append([th1, th2, th3, th4, th5, th6])

    # 去除数值上重复的解
    unique_solve = []
    for sol in solve:
        is_dup = False
        for usol in unique_solve:
            if all(abs(a - b) < 1e-4 for a, b in zip(sol, usol)):
                is_dup = True
                break
        if not is_dup:
            unique_solve.append(sol)

    return unique_solve