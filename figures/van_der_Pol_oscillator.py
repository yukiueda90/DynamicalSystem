import numpy as np
import matplotlib.pyplot as plt

# x'' - mu(1-x^2)x' + x = 0
# -> x' = y 
#    y' = mu(1-x^2)y - x
# ベクトル場を定義

mu = 1.0
def f(x: np.ndarray) -> np.ndarray:
  return np.array([
      x[1],
      mu * (1. - x[0]**2) * x[1] - x[0] 
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

fig, ax = plt.subplots(1, 2)
fig.tight_layout()
ax[0].set_aspect('equal')
ax[1].set_aspect('equal')
# optional: fix axis limits (important!)
ax[0].set_xlim(-2.6, 3.8)
ax[0].set_ylim(-3.6, 3.6)
ax[1].set_xlim(-2.6, 3.8)
ax[1].set_ylim(-3.6, 3.6)

# 時間区間分割
T: float = 100
N: int = 10000
t = np.linspace(0, T, N+1)
tau: float = T/N

# initial condition
x0 = 0.1 
y0 = 0.1 

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
ax[0].plot(x[0], x[1], color='tab:blue')

# initial condition
x0 = 3.1 
y0 = 3.1 

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
ax[1].plot(x[0], x[1], color='tab:blue')
plt.savefig('van_der_Pol_oscillator.png', bbox_inches='tight', dpi=300)
plt.show()
# ax.plot(X[0], X[1], linestyle='None', marker='o', color='tab:red')
# ax.set_xlim([-0.15, 1.75])
# ax.set_ylim([-0.9, 0.9])