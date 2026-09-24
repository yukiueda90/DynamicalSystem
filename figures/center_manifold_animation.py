import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# x' = y
# y' = -y - 3(x-1)^3

# ベクトル場を定義
def f(x: np.ndarray) -> np.ndarray:
  return np.array([
      x[1],
      -x[1] - 3*(x[0] - 1)**3
  ])

# 古典的ルンゲ=クッタ (４段陽的ルンゲ=クッタ)
def runge_kutta(x: np.ndarray, tau: float) -> np.ndarray:
    k1 = f(x)
    k2 = f(x + tau/2 * k1)
    k3 = f(x + tau/2 * k2)
    k4 = f(x + tau * k3)
    return x + tau/6 * (k1 + 2*k2 + 2*k3 + k4)

# 時間刻み 
tau: float = 5e-3 
N: int = 3000

# 初期条件
M: int = 19
theta = np.linspace(0, 2*np.pi, M+1)
x0 = 1 + 0.3 * np.cos(theta)
y0 = 0.3 * np.sin(theta)

# 配列を初期化
x = np.empty((M+1, 2, N+1), dtype = float) 
x[:, :, :] = np.nan
x[:, 0, 0] = x0
x[:, 1, 0] = y0

# 中心多様体の近似
phi = lambda x: -3 * x**3

# ルンゲ=クッタ法
for i in range(N):
  for j in range(M):
    x[j, :, i+1] = runge_kutta(x[j, :, i], tau)

# アニメーション作成の準備
fig, ax = plt.subplots()
fig.tight_layout()
ax.set_aspect('equal')
points = []
lines = []
# 描画範囲を固定
ax.set_xlim(0.65, 1.35)
ax.set_ylim(-0.32, 0.32)
# 中心多様体の近似のプロット
xc = np.linspace(-0.2, 0.2, 100)
line, = ax.plot(1.0 + xc, phi(xc), linewidth=5, color='tab:red')
lines.append(line)
for j in range(M):
  point, = ax.plot([], [], 'o', color='tab:blue')
  line, = ax.plot([], [], color='tab:blue')
  points.append(point) 
  lines.append(line)
# アニメーションによる更新を定義
def update(frame):
    for i in range(M):
        xn = x[i, :, frame]
        # 点のデータを追加
        points[i].set_data([xn[0]], [xn[1]])
        # 軌道のデータを追加
        lines[i+1].set_data(x[i, 0, :frame+1], x[i, 1, :frame+1])
    return

# アニメーション
step: int = 5
ani = FuncAnimation(fig, update, frames=range(0, x.shape[2], step), interval=10)
# ani.save("center_manifold.mp4", writer="ffmpeg", fps=60)

# プロット
plt.show()

