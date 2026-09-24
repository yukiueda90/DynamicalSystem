import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# x' = x(1-x^2-y^2) - y
# y' = y(1-x^2-y^2) + x 
# z' = -z

def f(x: np.ndarray) -> np.ndarray:
  return np.array([
      x[0] * (1-x[0]**2-x[1]**2) - x[1],
      x[1] * (1-x[0]**2-x[1]**2) + x[0], 
      -x[2]
  ])

# 古典的ルンゲ=クッタ (４段陽的ルンゲ=クッタ)
def runge_kutta(x: np.ndarray, tau: float) -> np.ndarray:
    k1 = f(x)
    k2 = f(x + tau/2 * k1)
    k3 = f(x + tau/2 * k2)
    k4 = f(x + tau * k3)
    return x + tau/6 * (k1 + 2*k2 + 2*k3 + k4)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
fig.tight_layout()
ax.set_aspect('equal') 

# 時間刻み
tau = 1e-2
N: int = 10000

# initial condition
for x0 in [[1.1, 1.1, 1.1], [-1.1, 1.1, -1.1], [1.1, -1.1, -1.1], [0.0, 0.0, -1.1]]:


  # 配列を初期化
  x = np.empty((3, N+1), dtype = float)
  x[:, :] = np.nan
  x[:, 0] = x0

  # ルンゲ=クッタ法
  for i in range(N):
    x[:, i+1] = runge_kutta(x[:, i], tau)

  # プロット
  ax.plot(x[0], x[1], x[2])

plt.savefig('stable_limit_cycle.png')
plt.show()
