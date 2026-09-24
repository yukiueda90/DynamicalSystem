import numpy as np
import matplotlib.pyplot as plt

# x' = -y + ax(x^2+y^2-1)(4-x^2-y^2),
# y' = x + ay(x^2+y^2-1)(4-x^2-y^2) 

# パラメータ 
a: float = 0.02

# 右辺のベクトル場
def f(x):
    s = x[0]**2 + x[1]**2 
    r1 = a * (s - 1) * (4 - s) 
    return np.array([
        -x[1] + x[0] * r1, 
        x[0] + x[1] * r1 
    ])
# ヤコビ行列
def Df(x): 
    s = x[0]**2 + x[1]**2 
    r1 = a * (s - 1) * (4 - s) 
    r2 = a * (5 - 2*s)
    return np.array([ 
        [r1 + 2 * x[0]**2 * r2, -1 + 2 * x[0] * x[1] * r2], 
        [1 + 2 * x[0] * x[1] * r2, r1 + 2 * x[1]**2 * r2]
    ])

# 古典的ルンゲ=クッタ (４段陽的ルンゲ=クッタ)
def runge_kutta(x: np.ndarray, M: np.ndarray, tau: float):
    k1 = f(x)
    K1 = Df(x) @ M
    k2 = f(x + tau/2 * k1)
    K2 = Df(x + tau/2 * k1) @ (M + tau/2 * K1)
    k3 = f(x + tau/2 * k2)
    K3 = Df(x + tau/2 * k2) @ (M + tau/2 * K2)
    k4 = f(x + tau * k3)
    K4 = Df(x + tau * k3) @ (M + tau * K3) 
    x = x + tau/6 * (k1 + 2*k2 + 2*k3 + k4) 
    M = M + tau/6 * (K1 + 2*K2 + 2*K3 + K4) 
    return x, M

# フローから x(T), M(T) を計算
def flow_map(q0: np.ndarray, time_steps: int = None): 
    x = q0[:-1]
    T = q0[-1]
    M = np.eye(2)
    if time_steps is None: 
        time_steps = 3000
    tau = T/time_steps 
    for _ in range(time_steps): 
        x, M = runge_kutta(x, M, tau) 
    return x, M

# ニュートン法を適用する写像
def F(q0: np.ndarray, x: np.ndarray): 
    # x は x, M = flow_map(q0) の出力を用いる
    return np.hstack((x - q0[:-1], q0[1]))

# プロット作成用にルンゲ=クッタ法を解く
def save_orbit(x0: np.ndarray, tau: float, N: int): 
    x = np.empty((2, N+1))
    x[:, 0] = x0
    for i in range(N):
        k1 = f(x[:, i])
        k2 = f(x[:, i] + tau/2 * k1)
        k3 = f(x[:, i] + tau/2 * k2)
        k4 = f(x[:, i] + tau * k3)
        x[:, i+1] = x[:, i] + tau/6 * (k1 + 2*k2 + 2*k3 + k4) 
    return x

# プロットの準備
fig, ax = plt.subplots() 
fig.tight_layout()
ax.set_aspect('equal')
tau: float = 1e-2 
N: int = 3000
# (比較用) 適当な初期値から軌道をプロット
x0 = [0.8, 0.0]
x = save_orbit(x0, tau, N) 
ax.plot(x[0], x[1], color='tab:red')
x0 = [1.2, 0.0]
x = save_orbit(x0, tau, N) 
ax.plot(x[0], x[1], color='tab:green')

# ============================================================
# 狙い撃ち法
# ============================================================

# 狙い撃ち法による反復の初期設定
q0 = np.array([
    1.05,   # r=1 に近い x0
    0,      # 位相条件より y0 = 0
    6.1     # 2\pi に近い T
])

# ニュートン法の反復
for _ in range(100):
    # x(T), M(T) の計算 
    x, M = flow_map(q0) 
    if np.linalg.norm(F(q0, x)) < 1e-10: 
        break
    # ヤコビ行列の計算 
    J = np.zeros((3, 3))
    J[:2, :2] = M - np.eye(2)
    J[:2, 2] = f(x)
    J[2, 0] = 0.0
    J[2, 1] = 1.0
    J[2, 2] = 0.0
    # ニュートン法による更新 
    d = np.linalg.solve(J, F(q0, x)) 
    q0 -= d

# 狙い撃ち法で求めた初期値から計算した軌道のプロット 
x = save_orbit(q0[:-1], tau, N) 
ax.plot(x[0], x[1], color='tab:blue')

plt.savefig('shooting_method.png')    
plt.show()

