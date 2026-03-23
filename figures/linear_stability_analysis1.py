import numpy as np
import matplotlib.pyplot as plt

# x' = -y + x(x^2+y^2)
# y' = x + y(x^2+y^2)

# ベクトル場を定義
def f(x: np.ndarray) -> np.ndarray:
  return np.array([
      -x[1] + x[0] * (x[0]**2 + x[1]**2),
      x[0] + x[1] * (x[0]**2 + x[1]**2)
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

# 時間区間分割
T: float = 30
N: int = 1000
t = np.linspace(0, T, N+1)
tau: float = T/N

# 初期条件
x0: float = 0.1
y0: float = 0

# 配列を初期化
x = np.empty((2, N+1), dtype = float)
x[:, 0] = [x0, y0]

# 古典的ルンゲ=クッタ (４段陽的ルンゲ=クッタ)
for i in range(N):
  x[:, i+1] = runge_kutta(x[:, i], tau)

# プロット
ax.plot(x[0], x[1])
# 向きを表す矢印
vec = f(x)
ax.quiver(x[0][50:N:50], x[1][50:N:50], vec[0][50:N:50], vec[1][50:N:50], scale=5, headwidth=3, minshaft=0, color='tab:blue')
ax.plot(x[0, 0], x[1, 0], linestyle='None', marker='o', label='initial value') # 初期値
ax.legend()
plt.show()

