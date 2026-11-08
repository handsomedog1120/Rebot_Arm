#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
笛卡尔3D空间(X/Y/Z)RRT路径规划器。

用于机械臂"自由空间转移"段（比如从当前位置移动到抓取点上方、
或抓取点上方移动到目标点上方）的避障路径规划：末端姿态固定不变，
只在X/Y/Z三维空间里做随机树搜索，绕开 OBSTACLES 里登记的障碍物包围盒，
返回一串笛卡尔坐标节点(含起止点)，交给上层按顺序用直线(MOVL)逐段连接执行。

★ OBSTACLES 目前是占位示例数据，请替换成现场实际的障碍物坐标/包围盒。
"""
import math
import random

from functions import workSpace, inv_k

# ──────────────────────────────────────────────────────────────────────────
# 工作空间边界，需与 functions.workSpace() 里的限位保持一致，作为随机采样范围
# ★ 已对齐 forensic_ws 验证版 functions.workSpace():
#   X[-0.20,0.30] Y[-0.15,0.30] Z[0.01,0.25]
# ──────────────────────────────────────────────────────────────────────────
WORKSPACE_BOUNDS = {
    'x': (-0.30, 0.30),
    'y': (-0.20, 0.30),
    'z': (-0.30, 0.25),
}

# ──────────────────────────────────────────────────────────────────────────
# ★ 障碍物占位示例数据 —— 请替换成现场真实物体的坐标/包围盒 ★
# 每个障碍物是一个轴对齐包围盒(AABB): {'name': ..., 'min': [x,y,z], 'max': [x,y,z]}
# min/max 单位为米，取值应落在 WORKSPACE_BOUNDS 范围内，坐标系与 workSpace() 一致。
# 不需要避障时可以把这个列表清空为 []，规划器会退化为"直接判断起止点连线是否可行"。
# ──────────────────────────────────────────────────────────────────────────
OBSTACLES = [
    {
        'name': 'example_evidence_stand',   # 示例：证物支架/物证台
        'min': [0.22, 0.00, 0.10],
        'max': [0.27, 0.05, 0.22],
    },
    {
        'name': 'example_camera_arm',       # 示例：相机支臂
        'min': [0.30, 0.15, 0.10],
        'max': [0.35, 0.20, 0.30],
    },
]

DEFAULT_MARGIN = 0.02              # 障碍物膨胀安全距离(米)，给执行器/夹爪体积留余量
DEFAULT_STEP_SIZE = 0.03           # 树扩展步长(米)
DEFAULT_MAX_ITER = 2000            # 最大采样迭代次数
DEFAULT_GOAL_SAMPLE_RATE = 0.1     # 直接采样终点的概率，加速收敛
DEFAULT_GOAL_TOLERANCE = 0.02      # 判定"足够接近终点"的距离阈值(米)
DEFAULT_COLLISION_RESOLUTION = 0.01  # 线段碰撞检测的采样间隔(米)


class _Node:
    __slots__ = ('point', 'parent')

    def __init__(self, point, parent=None):
        self.point = point
        self.parent = parent


def _distance(a, b):
    return math.sqrt(sum((a[k] - b[k]) ** 2 for k in range(3)))


def _random_point(bounds):
    return [
        random.uniform(*bounds['x']),
        random.uniform(*bounds['y']),
        random.uniform(*bounds['z']),
    ]


def _steer(from_pt, to_pt, step_size):
    """从 from_pt 朝 to_pt 方向前进最多 step_size 距离"""
    dist = _distance(from_pt, to_pt)
    if dist <= step_size:
        return list(to_pt)
    ratio = step_size / dist
    return [from_pt[k] + (to_pt[k] - from_pt[k]) * ratio for k in range(3)]


def _point_in_obstacle(point, obstacle, margin):
    lo = obstacle['min']
    hi = obstacle['max']
    return all(lo[i] - margin <= point[i] <= hi[i] + margin for i in range(3))


def is_state_valid(point, obstacles=OBSTACLES, margin=DEFAULT_MARGIN, check_ik=True):
    """
    检查一个笛卡尔坐标点是否是"合法状态"：
      1. 在 workSpace() 定义的矩形工作空间内
      2. 不落在任何(膨胀后的)障碍物包围盒内
      3. (可选) 逆运动学有解，即执行器物理上能到达该点，且满足关节限位
    """
    if not workSpace(point):
        return False
    for obs in obstacles:
        if _point_in_obstacle(point, obs, margin):
            return False
    if check_ik and not inv_k(point):
        return False
    return True


def segment_collision_free(p1, p2, obstacles=OBSTACLES, margin=DEFAULT_MARGIN,
                            resolution=DEFAULT_COLLISION_RESOLUTION, check_ik=True):
    """沿 p1->p2 等间隔采样，逐点做 is_state_valid 检查，只要有一点不合法就判定该线段不可行"""
    dist = _distance(p1, p2)
    steps = max(1, int(dist / resolution))
    for i in range(steps + 1):
        frac = i / steps
        pt = [p1[k] + (p2[k] - p1[k]) * frac for k in range(3)]
        if not is_state_valid(pt, obstacles, margin, check_ik):
            return False
    return True


def _extract_path(node):
    path = []
    while node is not None:
        path.append(node.point)
        node = node.parent
    path.reverse()
    return path


def _shortcut_path(path, obstacles=OBSTACLES, margin=DEFAULT_MARGIN, tries=100):
    """简单的路径捷径优化：随机尝试跳过中间节点直连，去掉RRT原始路径的锯齿感（非RRT*，够用即可）"""
    path = list(path)
    for _ in range(tries):
        if len(path) <= 2:
            break
        i = random.randint(0, len(path) - 2)
        j = random.randint(i + 1, len(path) - 1)
        if j - i <= 1:
            continue
        if segment_collision_free(path[i], path[j], obstacles, margin):
            path = path[:i + 1] + path[j:]
    return path


def plan_path(start_xyz, goal_xyz, obstacles=OBSTACLES, bounds=WORKSPACE_BOUNDS,
              step_size=DEFAULT_STEP_SIZE, max_iter=DEFAULT_MAX_ITER,
              goal_sample_rate=DEFAULT_GOAL_SAMPLE_RATE,
              goal_tolerance=DEFAULT_GOAL_TOLERANCE, margin=DEFAULT_MARGIN,
              smooth=True):
    """
    在笛卡尔3D空间对 start_xyz -> goal_xyz 做RRT路径搜索。

    返回：一串笛卡尔坐标节点组成的list（含起止点，例如 [[x0,y0,z0], [x1,y1,z1], ...]）。
    找不到可行路径时返回 None，由调用方决定兜底策略（例如退化为起止点直连）。
    """
    start_xyz = list(start_xyz)
    goal_xyz = list(goal_xyz)

    if not is_state_valid(start_xyz, obstacles, margin):
        return None
    if not is_state_valid(goal_xyz, obstacles, margin):
        return None

    # 起点终点之间本来就没有障碍，直接一条直线即可，不用跑树搜索
    if segment_collision_free(start_xyz, goal_xyz, obstacles, margin):
        return [start_xyz, goal_xyz]

    tree = [_Node(start_xyz, None)]

    for _ in range(max_iter):
        sample = goal_xyz if random.random() < goal_sample_rate else _random_point(bounds)
        nearest = min(tree, key=lambda n: _distance(n.point, sample))
        new_point = _steer(nearest.point, sample, step_size)

        if not is_state_valid(new_point, obstacles, margin):
            continue
        if not segment_collision_free(nearest.point, new_point, obstacles, margin):
            continue

        new_node = _Node(new_point, nearest)
        tree.append(new_node)

        if _distance(new_point, goal_xyz) <= goal_tolerance and \
                segment_collision_free(new_point, goal_xyz, obstacles, margin):
            goal_node = _Node(goal_xyz, new_node)
            path = _extract_path(goal_node)
            if smooth:
                path = _shortcut_path(path, obstacles, margin)
            return path

    return None
