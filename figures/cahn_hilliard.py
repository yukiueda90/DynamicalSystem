import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation 

# Allen-Cahn equation: 
# u_t = - (eps * u_xx + mu * u(1-u^2))_xx in [0, L]
# u(0, t) = 0, u(L, t) = 0 

# パラメータ 
eps: float = 0.1 
mu: float = 20.0

# 空間領域, 時間区間 
L: float = 1. 
T: float = 0.002 
M: int = 100 
N: int = 15000 
x = np.linspace(0, L, M+1) 
t = np.linspace(0, T, N+1) 
h: float = L/M 
tau: float = T/N 

# 離散ラプラシアン 
Lap: np.ndarray = -2 * np.eye(M+1) + np.eye(M+1, k=1) + np.eye(M+1, k=-1)
# 境界条件を考慮 
Lap[0, 1] = 2 
Lap[-1, -2] = 2
Lap = Lap / (h**2)

# 離散化された右辺 
def f(u: np.ndarray, u_prev: np.ndarray, h: float) -> np.ndarray: 
    u_mid = (u + u_prev) / 2 
    return -Lap @ (eps**2 * Lap @ u_mid + mu * u_mid * (1 - (u**2 + u_prev**2) / 2))
# ヤコビ行列 
def df(u: np.ndarray, u_prev: np.ndarray, h: float) -> np.ndarray: 
    DT = np.diag(3*(u**2) + 2*u*u_prev + u_prev**2)
    return -(eps**2)/2 * Lap @ Lap - mu/2 * Lap + mu/4 * Lap @ DT

# ニュートン法
def newton(x_prev: np.ndarray, tau: float, h: float, tol: float = 1e-6, max_iter: int = 30) -> np.ndarray: 
    x = x_prev 
    for _ in range(max_iter): 
        F = x - tau*f(x, x_prev, h) - x_prev 
        DF = np.eye(M+1) - tau*df(x, x_prev, h)
        x = x - np.linalg.solve(DF, F)
    if np.linalg.norm(np.linalg.solve(DF, F)) < tol:
       return x
    raise ValueError('Newton method did not converge.')

# 初期条件 (境界条件を満たすものを指定)
u0 = 0.7 * np.cos(3*np.pi*x/L) 
# 配列の初期化, 初期条件と境界条件の設定 
u = np.empty((M+1, N+1)) 
u[:, 0] = u0

# プロットの準備 
fig, ax = plt.subplots() 
ax.set_ylim([-1.1, 1.1])

# 初期条件のプロット 
line, = ax.plot(x, u[:, 0], color='tab:blue', linewidth=2.5) 
# plt.savefig('cahn_hilliard.pdf')

# アニメーションにおける update を定義 
def update(frame): 
    line.set_data(x, u[:, frame]) # 更新された u を入力 
    return 

# 計算 
for n in range(N): 
    u[:, n+1] = newton(u[:, n], tau, h) 

# プロット 
step: int = 30
ani = FuncAnimation(fig, update, frames=range(0, N, step), interval=5)
# ファイル出力する場合は下のコメントアウトを解除
ani.save("cahn_hilliard.mp4", writer="ffmpeg", fps=60)

plt.show()

