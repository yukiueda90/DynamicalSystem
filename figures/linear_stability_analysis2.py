import numpy as np
import matplotlib.pyplot as plt

# x' = y-x^2
# y' = -y

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

fig, ax = plt.subplots()
ax.set_aspect('equal')

# 中心多様体の近似
phi = lambda x: -3 * x**3
xc = np.linspace(-0.2, 0.2, 100)
ax.plot(1.0 + xc, phi(xc), linewidth=5, color='tab:red')

# 時間区間分割
T: float = 100
N: int = 1000
t = np.linspace(0, T, N+1)
tau: float = T/N

for theta in np.linspace(0, 2*np.pi, 20):
  # 初期条件
  x0: float = 1 + 0.3 * np.cos(theta)
  y0: float = 0.3 * np.sin(theta)

  # 配列を初期化
  x = np.empty((2, N+1), dtype = float)
  x[:, 0] = [x0, y0]

  # ルンゲ=クッタ法
  for i in range(N):
    x[:, i+1] = runge_kutta(x[:, i], tau)

  # プロット
  ax.plot(x[0], x[1], color='tab:blue')
  # # 向きを表す矢印
  # vec = f(x)
  # ax.quiver(x[0][5:30:5], x[1][5:30:5], vec[0][5:30:5], vec[1][5:30:5], scale=3, headwidth=3, minshaft=0, color='tab:blue')
  ax.plot(x[0, 0], x[1, 0], linestyle='None', marker='o', color='tab:orange') # 初期値

plt.show()
