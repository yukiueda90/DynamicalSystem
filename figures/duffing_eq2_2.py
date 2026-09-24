import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ダフィング方程式
# x' = y
# y' = x - x^3 + gamma cos(omega t)

gamma = 0.01
omega = 1.0

# 右辺
def f(t: float, x: np.ndarray):
    dx = np.empty_like(x)
    dx[..., 0] = x[..., 1]
    dx[..., 1] = x[..., 0] - x[..., 0]**3 + 0.01 * np.cos(t)
    return dx

# ルンゲ=クッタ法
def runge_kutta(t: float, x: np.ndarray, tau: float):
    k1 = f(t, x)
    k2 = f(t + tau/2, x + tau/2 * k1)
    k3 = f(t + tau/2, x + tau/2 * k2)
    k4 = f(t + tau, x + tau * k3)
    return x + tau/6 * (k1 + 2*k2 + 2*k3 + k4)

# 長方形上の点群
n: int = 300000
x0 = 0.1 * np.ones((4*n + 1, 2))
x0[0:n+1, 0] += np.linspace(0, 0.1, n+1)
x0[n+1:3*n+1, 0] += 0.1
x0[n:2*n+1, 1] += np.linspace(0, 0.1, n+1)
x0[2*n+1:, 1] += 0.1
x0[2*n:3*n+1, 0] += np.linspace(0, -0.1, n+1)
x0[3*n:, 1] += np.linspace(0, -0.1, n+1)
npts = len(x0)

# 時間刻み
tau = 5.0e-3
N = 4000

# アニメーションのフレーム生成時の時間刻み数
step = 10
# アニメーションのフレーム生成
def frame_generator():
    x = x0.copy()
    t = 0.0
    # 初期値
    yield x
    for k in range(N):
        x = runge_kutta(t, x, tau)
        t += tau
        if (k + 1) % step == 0:
            yield x
# アニメーション作成の準備
fig, ax = plt.subplots(figsize=(8, 5))
fig.tight_layout()
ax.set_aspect("equal")
ax.set_xlim(-2.1, 2.1)
ax.set_ylim(-1.1, 1.1)

line, = ax.plot([], [], color="tab:blue")

def init():
    line.set_data([], [])
    return line,


def update(x):
    line.set_data(x[:, 0], x[:, 1])
    return line,

# アニメーション作成
num_frames = N // step + 1
ani = FuncAnimation(
    fig,
    update,
    frames = frame_generator,
    init_func = init,
    interval = 10,
    blit = True,
    cache_frame_data = False,
    save_count = num_frames
)
# 出力
ani.save("test.mp4", writer="ffmpeg", fps=60)
# plt.show()