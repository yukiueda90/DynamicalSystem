import numpy as np 
import matplotlib.pyplot as plt 
from functools import partial 

# エネルギー: 
# H(x, y) = x^4/4 + y^2/2
# ハミルトン系: 
# x' = y 
# y' = -x^3
# 離散勾配法: 
# (x_{n+1} - x_n) / \tau = (y_{n+1} + y_n) / 2
# (y_{n+1} - y_n) / \tau = -(x_{n+1}^3 + x_{n+1}^2x_n + x_{n+1}x_n^2 + x_n^3) / 4
# Stormer-Verlet 法: 
# x_{n+1/2} = x_n + tau/2 * y_n 
# y_{n+1} = y_n - tau x_{n+1/2}^3 
# x_{n+1} = x_{n+1/2} + tau/2 * y_{n+1}

# エネルギー (ハミルトニアン)
H = lambda x: (x[0]**4) / 4 + (x[1]**2) / 2 

# ベクトル場を定義
def f(x: np.ndarray) -> np.ndarray:
    return np.array([
        x[1],
        -x[0]**3 
    ])

# V(x) = x^4/4 の微分 
dV = lambda x: x**3

# 古典的ルンゲ=クッタ (４段陽的ルンゲ=クッタ)
def runge_kutta(x: np.ndarray, tau: float) -> np.ndarray:
    k1 = f(x)
    k2 = f(x + tau/2 * k1)
    k3 = f(x + tau/2 * k2)
    k4 = f(x + tau * k3)
    return x + tau/6 * (k1 + 2*k2 + 2*k3 + k4)

# 離散勾配法から得られる関数およびそのヤコビ行列
def F(x: np.ndarray, x_n: np.ndarray, tau: float) -> np.ndarray:
    return x - x_n - tau * np.array([
        (x[1] + x_n[1]) / 2,
        -(x[0]**3 + x[0]**2 * x_n[0] + x[0] * x_n[0]**2 + x_n[0]**3) / 4 
    ])
def DF(x: np.ndarray, x_n: np.ndarray, tau: float) -> np.ndarray: 
    return np.array([
        [1, -tau/2], 
        [tau/4 * (3*x[0]**2 + 2*x_n[0]*x[0] + x_n[0]**2), 1] 
    ])

# ニュートン法 
def newton_method(Fn, DFn, x_init: np.ndarray) -> np.ndarray: 
    # 初期化
    x_prev = np.copy(x_init) # x^k
    x_update = np.empty_like(x_init) # x^{k+1}
    res: float = 1.0 # \|x^{k+1} - x^k\|
    count: int = 0 
    # 反復計算
    while res > 1e-10: 
      x_prev = x_update
      x_update = x_prev - np.linalg.solve(DFn(x_prev), Fn(x_prev)) 
      res = np.linalg.norm(x_update - x_prev)
      count += 1 
      if count == 1000:
         raise ValueError('Newton method does not converge.') 
    return x_update

# Stormer-Verlet 法 
def stormer_verlet(x: np.ndarray, tau: float) -> np.ndarray: 
    x[0] = x[0] + tau/2 * x[1] # x_{n+1/2}
    x[1] = x[1] - tau * dV(x[0]) 
    x[0] = x[0] + tau/2 * x[1] 
    return x 

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].set_aspect('equal')

# 時間区間分割
T: float = 1000
N: int = 3000
t = np.linspace(0, T, N+1)
tau: float = T/N
# tau の値を関数に部分適用 
F = partial(F, tau=tau) 
DF = partial(DF, tau=tau)

# 初期条件
x0 = [1.0, 0.0]

# 配列を初期化
x_RK = np.empty((2, N+1), dtype = float) # Runge-Kutta method
x_DG = np.empty((2, N+1), dtype = float) # Discrete Gradient method
x_SV = np.empty((2, N+1), dtype = float) # Stormer-Varlet method
x_RK[:] = np.nan
x_DG[:] = np.nan
x_SV[:] = np.nan
x_RK[:, 0] = x0
x_DG[:, 0] = x0
x_SV[:, 0] = x0

# ルンゲ=クッタ法 (比較用)
for i in range(N):
    x_RK[:, i+1] = runge_kutta(x_RK[:, i], tau)

# 離散勾配法 
for i in range(N): 
    # x_n を部分適用 
    Fn = partial(F, x_n = x_DG[:, i])
    DFn = partial(DF, x_n = x_DG[:, i])
    # ニュートン法
    x_DG[:, i+1] = newton_method(Fn, DFn, x_DG[:, i])

# Stormer-Varlet 法 
for i in range(N): 
    x_SV[:, i+1] = stormer_verlet(x_SV[:, i], tau)

# プロット
ax[0].plot(x_RK[0], x_RK[1], label='Runge-Kutta')
ax[0].plot(x_DG[0], x_DG[1], label='Discrete Gradient Method')
ax[0].plot(x_SV[0], x_SV[1], label='Stormer-Verlet')
ax[0].legend()
ax[1].plot(t, H(x_RK))
ax[1].plot(t, H(x_DG))
ax[1].plot(t, H(x_SV))
fig.tight_layout()
plt.show()



