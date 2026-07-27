import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

# 2D Lotka–Volterra equation: 
# x' = x(a-by), 
# y' = y(cx-d), 
a: float = 1.0 
b: float = 1.0 
c: float = 1.5 
d: float = 1.0

# ベクトル場を定義
def f(x: np.ndarray) -> np.ndarray:
  return np.array([
      x[0] * (a - b*x[1] + 0.05*x[0]),
      x[1] * (c*x[0] - d)
  ])


# 平衡点
X = np.array([0.0, 0.0])

# 古典的ルンゲ=クッタ (４段陽的ルンゲ=クッタ)
def runge_kutta(x: np.ndarray, tau: float) -> np.ndarray:
    k1 = f(x)
    k2 = f(x + tau/2 * k1)
    k3 = f(x + tau/2 * k2)
    k4 = f(x + tau * k3)
    return x + tau/6 * (k1 + 2*k2 + 2*k3 + k4)

fig, ax = plt.subplots()

# initial plot (empty)
point, = ax.plot([], [], 'o', color='tab:blue')
line, = ax.plot([], [], color='tab:blue')

# 時間区間分割
T: float = 100.0
N: int = 10000
t = np.linspace(0, T, N+1)
tau: float = T/N

x0 = 0.5
y0 = 0.1 


# 配列を初期化
x = np.empty((2, N+1), dtype = float)
x[:, :] = np.nan
x[:, 0] = [x0, y0]

# ルンゲ=クッタ法
for i in range(N):
  x[:, i+1] = runge_kutta(x[:, i], tau)
  if np.linalg.norm(x[:, i+1] - X) > 500.0:
    break

# optional: fix axis limits (important!)
ax.set_xlim(-0.1, 10.1)
ax.set_ylim(-0.1, 10.1)
# ax.set_position([0, 0, 1, 1])

def update(frame):
    xn = x[:, frame]

    # wrap scalars as sequences
    point.set_data([xn[0]], [xn[1]])

    line.set_data(x[0, :frame+1], x[1, :frame+1])

    return point, line

step: int = 5
ani = FuncAnimation(fig, update, frames=range(0, x.shape[1], step), interval=10)
# ani.save("lotka_volterra_2d.mp4", writer="ffmpeg", fps=60, savefig_kwargs={"bbox_inches": "tight", "pad_inches": 0})

plt.show()

