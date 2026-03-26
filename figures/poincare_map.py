import numpy as np
import matplotlib.pyplot as plt

# x'' - mu(1-x^2)x' + x = 0
# -> x' = y 
#    y' = mu(1-x^2)y - x
# ベクトル場を定義

mu = 0.5
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
ax[0].set_xlim(-2.6, 2.6)
ax[0].set_ylim(-3.2, 3.2)
ax[1].set_xlim(0.0, 2.1)
ax[1].set_ylim(-0.5, 0.5)

ax[1].set_xlim(2.00175, 2.0026)
ax[1].set_ylim(-0.0004, 0.0004)
# ax[1].set_xticks([])
ax[1].set_yticks([])


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
ax[0].plot([0, 3], [0, 0], color='tab:orange')
ax[0].plot(x[0], x[1], color='tab:blue')
ax[0].plot([1.88, 1.88, 2.12, 2.12, 1.88], [-0.1, 0.1, 0.1, -0.1, -0.1], color='k')
ax[0].text(0.15, 0.05, '$\mathbf{x}$')
ax[0].text(0.65, 0.10, '$P(\mathbf{x})$')
ax[0].text(1.35, 0.10, '$P^2(\mathbf{x})$')
ax[0].plot([0.15, 0.71, 1.72], [0, 0, 0], linestyle='None', marker='o', markersize=3, color='k')

# プロット
ax[1].plot([0, 3], [0, 0], color='tab:orange')
ax[1].plot(x[0], x[1], color='tab:blue')
ax[1].text(2.00187, 0.00002, '$P^3(\mathbf{x})$')
ax[1].text(2.00235, 0.00002, '$P^4(\mathbf{x})$')
ax[1].plot([2.00186, 2.002438, 2.00246, 2.00247, 2.00248], [0, 0, 0, 0, 0], linestyle='None', marker='o', markersize=3, color='k')

plt.savefig('poincare_map.png', bbox_inches='tight', dpi=300)
plt.show()
# ax.plot(X[0], X[1], linestyle='None', marker='o', color='tab:red')
# ax.set_xlim([-0.15, 1.75])
# ax.set_ylim([-0.9, 0.9])