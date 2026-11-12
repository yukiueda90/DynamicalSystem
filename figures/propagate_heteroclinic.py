import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation

# パラメータ 
k: float = 1.0 
r: float = 5.0

# 空間領域, 時間区間
L: float = 50.0 
T: float = 5.0 
M: int = 500 
N: int = 1000 
x = np.linspace(-L/2, L/2, M+1) 
# t = np.linspace(0, T, N+1)
# h: float = L/M 
# tau: float = T/N 
# # m と離散サイン変換の行列
# m = np.arange(1, M)
# S = 2/M * np.sin(np.pi * np.outer(m, m) / M)

fig, ax = plt.subplots(1, 2, figsize=(12, 4)) 

# Ablowitz-Zeppetella solution 
speed = 5/np.sqrt(6) * np.sqrt(k*r)

u = lambda z: 1/(1+np.exp(np.sqrt(r/(6*k))*z))**2
ax[0].plot(x, u(x), linewidth=2)

# heteroclinic orbit 
def func(x): 
    return np.array([
        x[1],
        -speed/k*x[1] - r/k*x[0]*(1. - x[0])
    ])
def RungeKutta(x, tau): 
    # Runge-Kutta
    k1 = func(x) 
    k2 = func(x + tau/2*k1) 
    k3 = func(x + tau/2*k2) 
    k4 = func(x + tau*k3) 
    return x + tau/6 * (k1 + 2*k2 + 2*k3 + k4) 
def orbit(x0, tau, N):
    x = np.empty((2, N)) 
    x[:, 0] = x0
    for n in range(N-1):
        x[:, n+1] = RungeKutta(x[:, n], tau) 
    return x 

# test 
x0_list = np.array([ 
    [0.995, 0.0], 
])
for x0 in x0_list:
    x = orbit(x0, 1e-2, 1000)
    ax[1].plot(x[0], x[1], color='tab:blue')
    idx = np.arange(400, 800, 200) # 矢印の位置
    vec = func(x[:, idx]) # 矢印の向き
    norm = np.linalg.norm(vec, axis=0)
    vec = vec / norm # 矢印の大きさを正規化
    ax[1].quiver(x[0, idx], x[1, idx], vec[0], vec[1], scale=30, headwidth=3, minshaft=0, color='tab:blue')
x0_list = np.array([ 
    [0.8, 0.5], 
    [0.5, 0.5],  
    [0.2, 0.5],  
    [-0.4, -0.3],  
    [-0.2, -0.3],  
    [0.8, -0.3],  
])
for x0 in x0_list:
    x = orbit(x0, 1e-2, 1000)
    ax[1].plot(x[0], x[1], color='tab:blue')
    idx = np.arange(10, 200, 250) # 矢印の位置
    vec = func(x[:, idx]) # 矢印の向き
    norm = np.linalg.norm(vec, axis=0)
    vec = vec / norm # 矢印の大きさを正規化
    ax[1].quiver(x[0, idx], x[1, idx], vec[0], vec[1], scale=30, headwidth=3, minshaft=0, color='tab:blue')
x0_list = np.array([ 
    [1.01, 0.01],
])
for x0 in x0_list:
    x = orbit(x0, 1e-2, 350)
    ax[1].plot(x[0], x[1], color='tab:blue')
    idx = np.arange(200, 300, 100) # 矢印の位置
    vec = func(x[:, idx]) # 矢印の向き
    norm = np.linalg.norm(vec, axis=0)
    vec = vec / norm # 矢印の大きさを正規化
    ax[1].quiver(x[0, idx], x[1, idx], vec[0], vec[1], scale=30, headwidth=3, minshaft=0, color='tab:blue')
x0_list = np.array([
    [0.91, 0.5], 
    [1.2, -0.3],
    [1.05, 0.5],
    [1.055, -0.3]
])
for x0 in x0_list:
    x = orbit(x0, 1e-2, 100)
    ax[1].plot(x[0], x[1], color='tab:blue')
    idx = np.arange(20, 100, 100) # 矢印の位置
    vec = func(x[:, idx]) # 矢印の向き
    norm = np.linalg.norm(vec, axis=0)
    vec = vec / norm # 矢印の大きさを正規化
    ax[1].quiver(x[0, idx], x[1, idx], vec[0], vec[1], scale=30, headwidth=3, minshaft=0, color='tab:blue')

ax[1].plot([0.], [0.], color='tab:green', linestyle='None', marker='o', markersize=8)
ax[1].plot([1.], [0.], color='tab:red', linestyle='None', marker='o', markersize=8)
ax[1].set_xlim([-0.4, 1.3])
ax[1].set_ylim([-0.3, 0.5])

fig.tight_layout()
plt.savefig('propagate_heteroclinic.png')
plt.show()