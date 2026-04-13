import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


# x' = -x
# y' = z

# ベクトル場を定義
def f(x: np.ndarray) -> np.ndarray:
  return np.array([
      -x[0],
      x[2],  
      0.0
  ])

# 古典的ルンゲ=クッタ (４段陽的ルンゲ=クッタ)
def runge_kutta(x: np.ndarray, tau: float) -> np.ndarray:
    k1 = f(x)
    k2 = f(x + tau/2 * k1)
    k3 = f(x + tau/2 * k2)
    k4 = f(x + tau * k3)
    return x + tau/6 * (k1 + 2*k2 + 2*k3 + k4)

# 時間区間分割
T: float = 8
N: int = 5000
t = np.linspace(0, T, N+1)
tau: float = T/N

# 初期条件
M: int = 19
theta = np.linspace(0, 2*np.pi, M+1)
x0 = 1.8 * np.cos(theta)
y0 = 1.8 * np.sin(theta)
z0 = 0.1 * np.sin(theta)

# 配列を初期化
x = np.empty((M+1, 3, N+1), dtype = float) 
x[:, :, :] = np.nan
x[:, 0, 0] = x0
x[:, 1, 0] = y0
x[:, 2, 0] = z0

# 中心多様体の近似
# phi = lambda x: -3 * x**3

# ルンゲ=クッタ法
for i in range(N):
  for j in range(M):
    x[j, :, i+1] = runge_kutta(x[j, :, i], tau)

# fig, ax = plt.subplots()
fig = plt.figure() 
ax = fig.add_subplot(111, projection='3d')
fig.tight_layout()
ax.set_aspect('equal')
# optional: fix axis limits (important!)
ax.set_xlim3d(-2.2, 2.2)
ax.set_ylim3d(-2.2, 2.2)
ax.set_zlim3d(-2.2, 2.2)

# initial plot (empty)
points = []
lines = []
stable = np.linspace(-2.2, 2.2, 100)
line, = ax.plot(stable, np.zeros_like(stable), np.zeros_like(stable), linewidth=5, color='tab:green')
lines.append(line)
for j in range(M):
  point, = ax.plot([], [], [], 'o', color='tab:blue')
  line, = ax.plot([], [], [], color='tab:blue')
  points.append(point) 
  lines.append(line)
# define update in animation
def update(frame):
  for i in range(M):
    xn = x[i, :, frame]
    # update point
    points[i].set_data([xn[0]], [xn[1]])
    points[i].set_3d_properties([xn[2]])
    # update trajectory
    lines[i+1].set_data(x[i, 0, :frame+1], x[i, 1, :frame+1])
    lines[i+1].set_3d_properties(x[i, 2, :frame+1])
    # artists.extend([points[i], lines[i+1]])
  return
    # return artists

step: int = 15
ani = FuncAnimation(fig, update, frames=range(0, x.shape[2], step), interval=3)
ani.save("animation.mp4", writer="ffmpeg", fps=60)

# プロット
# ax.plot(x[0], x[1], color='tab:blue')
plt.show()

