"""
绘制函数曲线及其导数
- f(x)    : 原函数
- f'(x)   : 一阶导数
- f''(x)  : 二阶导数
"""

import numpy as np
import matplotlib.pyplot as plt

# ── 1. 设置中文字体和负号显示 ──────────────────────────────────────────────
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ── 2. 定义函数（可以自由修改）─────────────────────────────────────────────
def f(x):
    """原函数：这里用 f(x) = 10x^3 - 15x^4 + 6x^5 作为示例"""
    return 10 * x**3 - 15 * x**4 + 6 * x**5

def df(x):
    """一阶导数（解析解）"""
    return 30 * x**2 - 60 * x**3 + 30 * x**4

def ddf(x):
    """二阶导数（解析解）"""
    return 60 * x - 180 * x**2 + 120 * x**3

# ── 3. 数值求导（如果你懒得手写解析解，可以用这个）────────────────────────
def numerical_derivative(func, x, h=1e-5):
    """中心差分法求数值导数"""
    return (func(x + h) - func(x - h)) / (2 * h)

# ── 4. 生成数据 ────────────────────────────────────────────────────────────
x = np.linspace(0,1, 500)
y = f(x)
dy = df(x)
ddy = ddf(x)

# 如果想验证数值导数：
# dy_num = numerical_derivative(f, x)

# ── 5. 绘图 ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# 子图 1：原函数
axes[0].plot(x, y, color="tab:blue", linewidth=2, label="f(x)")
axes[0].set_ylabel("f(x)")
axes[0].set_title("函数及其导数")
axes[0].legend(loc="upper right")
axes[0].grid(True, alpha=0.3)
axes[0].axhline(0, color="gray", linewidth=0.8, linestyle="--")

# 子图 2：一阶导数
axes[1].plot(x, dy, color="tab:orange", linewidth=2, label="f'(x)")
axes[1].set_ylabel("f'(x)")
axes[1].legend(loc="upper right")
axes[1].grid(True, alpha=0.3)
axes[1].axhline(0, color="gray", linewidth=0.8, linestyle="--")

# 子图 3：二阶导数
axes[2].plot(x, ddy, color="tab:green", linewidth=2, label="f''(x)")
axes[2].set_ylabel("f''(x)")
axes[2].set_xlabel("x")
axes[2].legend(loc="upper right")
axes[2].grid(True, alpha=0.3)
axes[2].axhline(0, color="gray", linewidth=0.8, linestyle="--")

plt.tight_layout()
plt.savefig("function_and_derivatives.png", dpi=150)
plt.show()