import numpy as np
import matplotlib.pyplot as plt

# x' = y
# y' = x - x**3
# ベクトル場を定義
def f(x: np.ndarray) -> np.ndarray:
  return np.array([
      x[1],
      x[0] - x[0]**3
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
ax.set_aspect('equal')

# 時間区間分割
T: float = 10
N: int = 3000
t = np.linspace(0, T, N+1)
tau: float = T/N

theta = np.linspace(np.pi/6, 1/3 * np.pi, 2)
# theta = np.append(theta, 0.025 + theta[28] )
# theta = np.append(theta, 0.033 + theta[13] )
xinit = X[0] + 0.01 * np.cos(theta)
yinit = X[1] + 0.01 * np.sin(theta)
# xinit = np.append(xinit, X[0]+0.01*np.cos(theta[29] + np.pi/2))
# yinit = np.append(yinit, X[1]+0.01*np.sin(theta[29] + np.pi/2))
# xinit = np.append(xinit, X[0]+0.01*np.cos(theta[31] + np.pi/2))
# yinit = np.append(yinit, X[1]+0.01*np.sin(theta[31] + np.pi/2))

for k, (x0, y0) in enumerate(zip(xinit, yinit)):

  # 配列を初期化
  x = np.empty((2, N+1), dtype = float)
  x[:, :] = np.nan
  x[:, 0] = [x0, y0]

  # ルンゲ=クッタ法
  for i in range(N):
    x[:, i+1] = runge_kutta(x[:, i], tau)
    if np.linalg.norm(x[:, i+1] - X) > 10.0:
      break

  # プロット
  # if k == 30 or k == 31:
  #     ax.plot(x[0], x[1], color='tab:green', linewidth=5)
  # if k == 32 or k == 33:
  #     ax.plot(x[0], x[1], color='tab:red', linewidth=5)
  # if k != 30 and k!= 31:
  #     ax.plot(x[0], x[1], color='tab:blue')
  ax.plot(x[0], x[1], color='tab:blue')
  # ax.plot(X[0], X[1], linestyle='None', marker='o', color='tab:red')
  ax.set_xlim([-0.15, 1.75])
  ax.set_ylim([-0.9, 0.9])

  # # 向きを表す矢印
  vec = f(x)
  # if k != 30 and k!= 31:
  ax.quiver(x[0][0:3000:500], x[1][0:3000:500], vec[0][0:3000:500], vec[1][0:3000:500], scale=50, headwidth=3, minshaft=0, color='tab:blue')
  # ax.plot(x[0, 0], x[1, 0], linestyle='None', marker='o', color='tab:orange') # 初期値

plt.show()