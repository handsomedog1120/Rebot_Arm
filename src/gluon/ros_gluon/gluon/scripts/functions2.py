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

def normalize_angle(angle, limit=2.79):
    """将角度归一化到 [-limit, limit] 范围内，选择绝对值最小的有效解"""
    # 先归一化到 [-pi, pi]
    while angle > pi:
        angle -= 2*pi
    while angle < -pi:
        angle += 2*pi
    
    # 生成候选解（原角度加减 2pi）
    candidates = [angle]
    if angle + 2*pi <= limit:
        candidates.append(angle + 2*pi)
    if angle - 2*pi >= -limit:
        candidates.append(angle - 2*pi)
    
    # 筛选在限位内的解
    valid = [c for c in candidates if -limit <= c <= limit]
    
    if valid:
        return min(valid, key=abs)  # 返回最接近0的解
    return None

def workSpace(pos):
    '''划定工作区域'''
    x = pos[0]
    y = pos[1]
    z = pos[2]
    xmin = -0.80
    xmax = 0.80
    ymin = -0.80
    ymax = 0.80
    zmin = -0.40
    zmax = 0.40
    if xmin <= x <= xmax and ymin <= y <= ymax and zmin <= z <= zmax:
        return True
    else:
        print(pos, 'out of space')
        return False

def inv_k(target_pos, target_orn=None):
    """
    UR3/5/10, gluon通用的求解方法（修复版）
    支持左右手解和th4内外转

    @Returns
    solve: [[th1, th2, th3, th4, th5, th6], ...] 多组解
    """
    solve = []
    if target_orn is None:
        target_orn = [0, 0, 0, 1]
    
    x, y, z = target_pos[0], target_pos[1], target_pos[2]
    qx, qy, qz, qw = target_orn[0], target_orn[1], target_orn[2], target_orn[3]

    # 四元数转旋转矩阵
    r = Quaternion_2_matrix([qx, qy, qz, qw])

    # th1 - 左右手两解
    a1 = x**2 + y**2 - 0.00627264
    if a1 < -0.0001:  # 无实数解
        return []
    
    # 处理数值误差
    if abs(a1) < 0.0001:
        a1 = 0
    
    sqrt_a1 = sqrt(max(a1, 0))
    atan2_xy = arctan2(y, x)
    d = 0.0792
    
    # 左右手两解
    th1_candidates = []
    if sqrt_a1 < 0.0001:  # 特殊情况：a1≈0
        th1_candidates = [-atan2_xy]
    else:
        th1_right = arctan2(sqrt_a1, d) - atan2_xy   # 右手
        th1_left = -arctan2(sqrt_a1, d) - atan2_xy   # 左手
        th1_candidates = [th1_right, th1_left]
    
    for th1 in th1_candidates:
        # 检查th1限位
        if not (-2.79 < th1 < 2.79):
            continue
            
        s1, c1 = sin(th1), cos(th1)
        
        # th5 - 正负两解
        val = c1*r[0, 0] - s1*r[1, 0]
        # 裁剪到有效范围
        val = max(-1.0, min(1.0, val))
        
        if abs(abs(val) - 1.0) < 1e-6:  # 边界情况
            th5_candidates = [0.0 if val > 0 else pi]
        else:
            th5_base = arccos(val)
            th5_candidates = [th5_base, -th5_base]
        
        for th5 in th5_candidates:
            # 检查th5限位
            th5_norm = normalize_angle(th5, limit=2.79)
            if th5_norm is None:
                continue
            th5 = th5_norm
            
            s5, c5 = sin(th5), cos(th5)
            
            # th6 - 根据th5的符号有不同解
            px = r[2, 2]*s5
            py = r[2, 1]*s5
            d_val = r[2, 0]*c5
            
            # 处理数值误差
            px = 0 if abs(px) < 1e-10 else px
            py = 0 if abs(py) < 1e-10 else py
            d_val = 0 if abs(d_val) < 1e-10 else d_val
            
            denom = sqrt(max(px**2 + py**2 - d_val**2, 0))
            
            if denom < 1e-10:  # 奇异点
                th6_candidates = [0.0]
            else:
                base = -arctan2(d_val, denom) - arctan2(py, px)
                th6_base = base
                th6_alt = pi - base if base > 0 else -pi - base
                
                # 归一化
                for t in [th6_base, th6_alt]:
                    while t > pi:
                        t -= 2*pi
                    while t < -pi:
                        t += 2*pi
                
                th6_candidates = [th6_base, th6_alt]
            
            for th6 in th6_candidates:
                # 检查th6限位
                if not (-2.79 < th6 < 2.79):
                    continue
                    
                s6, c6 = sin(th6), cos(th6)
                
                # 计算腕部位置
                wx = x - 0.08*(r[0, 2]*c6 - r[0, 1]*s6)
                wy = y - 0.08*(r[1, 2]*c6 - r[1, 1]*s6)
                wz = z - 0.08*(r[2, 2]*c6 - r[2, 1]*s6)
                
                # th3 - 肘部角度
                D = wx**2 + wy**2 + (wz - 0.105)**2 - 0.06656121
                
                if D > 0.06083072 or D < -0.06083072:
                    continue
                
                # 裁剪到有效范围
                D = max(-0.06083072, min(0.06083072, D))
                
                # th3有两解：肘部向上和向下（但代码注释说没有"上下手"，只取一解）
                th3_base = arccos(-D/0.06083072) - pi
                
                for th3 in [th3_base]:  # 如需上下手解，改为 [th3_base, -th3_base - 2*pi/2]
                    # th2 - 肩部角度
                    tmp = D + 0.06083072
                    if tmp < 1e-10:
                        continue
                    
                    dist_w = sqrt(tmp)
                    if dist_w < 1e-10:
                        continue
                    
                    val2 = (wz - 0.105) / dist_w
                    val2 = max(-1.0, min(1.0, val2))
                    
                    th_tmppe = arccos(val2)
                    th2 = th_tmppe + th3/2
                    
                    # 检查th2限位
                    if not (0 < th2 < 1.65):
                        continue
                    
                    # th4 - 关键修复：考虑内外转两解
                    # 计算 th_tmpe2 (th2+th3+th4)
                    A = s1*(r[0, 2]*c6 - r[0, 1]*s6) + c1*(r[1, 2]*c6 - r[1, 1]*s6)
                    B = r[2, 2]*c6 - r[2, 1]*s6
                    
                    th_tmpe2_base = arctan2(A, B)
                    
                    # th_tmpe2 的两种配置（相差pi），对应th4内外转
                    for delta in [0, pi]:
                        th_tmpe2 = th_tmpe2_base + delta
                        
                        # 归一化到 [-pi, pi]
                        while th_tmpe2 > pi:
                            th_tmpe2 -= 2*pi
                        while th_tmpe2 < -pi:
                            th_tmpe2 += 2*pi
                        
                        th4 = th_tmpe2 - th2 + th3
                        
                        # 归一化th4到 [-pi, pi]
                        while th4 > pi:
                            th4 -= 2*pi
                        while th4 < -pi:
                            th4 += 2*pi
                        
                        # 检查th4限位
                        if not (-2.79 < th4 < 2.79):
                            continue
                        
                        # 验证正运动学
                        c234 = cos(th_tmpe2)
                        s234 = sin(th_tmpe2)
                        
                        # 计算验证误差
                        err_x = c1*c5 - s1*c234*s5 - r[0, 0]
                        err_y = s1*c5 + c1*c234*s5 + r[1, 0]  # 注意符号
                        err_z = s234 - r[2, 0]
                        
                        dist = err_x**2 + err_y**2 + err_z**2
                        
                        if dist < 1e-5:
                            solution = [th1, th2, th3, th4, th5, th6]
                            # 避免重复解
                            is_new = True
                            for existing in solve:
                                if all(abs(a - b) < 1e-4 for a, b in zip(existing, solution)):
                                    is_new = False
                                    break
                            if is_new:
                                solve.append(solution)
    
    return solve


# 测试代码
if __name__ == "__main__":
    # 测试您提到的案例
    test_pos = [0.3, 0, 0.2]
    test_orn = [0, 0, 0, 1]  # 单位四元数
    
    print(f"测试位置: {test_pos}")
    print(f"测试姿态: {test_orn}")
    print("-" * 50)
    
    solutions = inv_k(test_pos, test_orn)
    
    print(f"找到 {len(solutions)} 组解:\n")
    for i, sol in enumerate(solutions):
        print(f"解 {i+1}:")
        print(f"  th1 = {sol[0]:.4f} rad ({np.degrees(sol[0]):.2f}°)")
        print(f"  th2 = {sol[1]:.4f} rad ({np.degrees(sol[1]):.2f}°)")
        print(f"  th3 = {sol[2]:.4f} rad ({np.degrees(sol[2]):.2f}°)")
        print(f"  th4 = {sol[3]:.4f} rad ({np.degrees(sol[3]):.2f}°) {'(向内)' if sol[3] < 0 else '(向外)'}")
        print(f"  th5 = {sol[4]:.4f} rad ({np.degrees(sol[4]):.2f}°)")
        print(f"  th6 = {sol[5]:.4f} rad ({np.degrees(sol[5]):.2f}°)")
        print()