import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation 

# Allen-Cahn equation: 
# u_t = eps * u_xx + mu * u(1-u^2) in [0, L]
# u(0, t) = 0, u(L, t) = 0 

# パラメータ 
eps: float = 0.1 
mu: float = 20.0

# 2 階中心差分による空間離散化 
def f(u: np.ndarray, h: float) -> np.ndarray: 
    vec = np.empty_like(u) 
    vec[0] = 0. 
    vec[-1] = 0. 
    vec[1:-1] = eps**2 * (u[2:] - 2*u[1:-1] + u[:-2]) / (h**2) + mu * u[1:-1]*(1. - u[1:-1]**2) 
    return vec

# 古典的ルンゲ=クッタ (４段陽的ルンゲ=クッタ)
def runge_kutta(u: np.ndarray, tau: float, h: float) -> np.ndarray:
    k1 = f(u, h)
    k2 = f(u + tau/2 * k1, h)
    k3 = f(u + tau/2 * k2, h)
    k4 = f(u + tau * k3, h)
    return u + tau/6 * (k1 + 2*k2 + 2*k3 + k4) 

# 空間領域, 時間区間 
L: float = 1. 
T: float = 0.3 
M: int = 300 
N: int = 5000 
x = np.linspace(0, L, M+1) 
t = np.linspace(0, T, N+1) 
h: float = L/M 
tau = float = T/N 

# 初期条件 (境界条件を満たすものを指定)
u0 = 0.6 * np.sin(3*np.pi*x/L) 
# 配列の初期化, 初期条件と境界条件の設定 
u = np.empty((M+1, N+1)) 
u[:, 0] = u0
u[0, :] = 0 
u[-1, :] = 0

# プロットの準備 
fig, ax = plt.subplots() 
ax.set_ylim([-1.1, 1.1])

# 初期条件のプロット 
line, = ax.plot(x, u[:, 0], color='tab:blue') 
# アニメーションにおける update を定義 
def update(frame): 
    line.set_data(x, u[:, frame]) # 更新された u を入力 
    return 

# 計算 
for n in range(N): 
    u[:, n+1] = runge_kutta(u[:, n], tau, h) 
    # u[0, n+1] = 0 # f[0] = 0 のため不要
    # u[-1, n+1] = 0 # f[-1] = 0 のため不要

# プロット 
step: int = 10
ani = FuncAnimation(fig, update, frames=range(0, N, step), interval=5)
# ファイル出力する場合は下のコメントアウトを解除
ani.save("allen_cahn.mp4", writer="ffmpeg", fps=60)

plt.show()

