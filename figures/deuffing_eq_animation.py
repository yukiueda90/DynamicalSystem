import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# x' = y
# y' = x - x**3
# ベクトル場を定義
def f(x: np.ndarray, t: np.ndarray) -> np.ndarray:
  return np.array([
      x[1],
      x[0] - x[0]**3 + 0.01 * np.cos(t)
  ])


# 平衡点
X = np.array([0.0, 0.0])

# 古典的ルンゲ=クッタ (４段陽的ルンゲ=クッタ)
def runge_kutta(x: np.ndarray, t: float, tau: float) -> np.ndarray:
    k1 = f(x, t)
    k2 = f(x + tau/2 * k1, t + tau/2)
    k3 = f(x + tau/2 * k2, t + tau/2)
    k4 = f(x + tau * k3, t + tau)
    return x + tau/6 * (k1 + 2*k2 + 2*k3 + k4)

fig, ax = plt.subplots()
fig.tight_layout()
ax.set_aspect('equal')

# initial plot (empty)
point, = ax.plot([], [], 'o', color='tab:blue')
line, = ax.plot([], [], color='tab:blue')

# 時間区間分割
T: float = 60
N: int = 3000
t = np.linspace(0, T, N+1)
tau: float = T/N

x0 = X[0] + 0.01 * np.cos(np.pi/6)
y0 = X[1] + 0.01 * np.sin(np.pi/6)


# 配列を初期化
x = np.empty((2, N+1), dtype = float)
x[:, :] = np.nan
x[:, 0] = [x0, y0]

# ルンゲ=クッタ法
for i in range(N):
  x[:, i+1] = runge_kutta(x[:, i], t[i], tau)
  if np.linalg.norm(x[:, i+1] - X) > 10.0:
    break

# optional: fix axis limits (important!)
ax.set_xlim(-2.1, 2.1)
ax.set_ylim(-1.1, 1.1)

def update(frame):
    xn = x[:, frame]
    # wrap scalars as sequences
    point.set_data([xn[0]], [xn[1]])
    line.set_data(x[0, :frame+1], x[1, :frame+1])
    return point, line

step: int = 5
ani = FuncAnimation(fig, update, frames=range(0, x.shape[1], step), interval=10)
ani.save("animation.mp4", writer="ffmpeg", fps=60)

# プロット
# ax.plot(x[0], x[1], color='tab:blue')
plt.show()
# ax.plot(X[0], X[1], linestyle='None', marker='o', color='tab:red')
# ax.set_xlim([-0.15, 1.75])
# ax.set_ylim([-0.9, 0.9])
