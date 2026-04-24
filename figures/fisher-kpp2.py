import numpy as np
from scipy.fft import dst, idst
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
x = np.linspace(0, L, M+2) 
t = np.linspace(0, T, N+1)
h: float = L/(M+1) 
tau: float = T/N 
# m と離散サイン変換の行列
m = np.arange(1, M+1)
S = np.sin(np.pi * np.outer(m, m) / (M+1))


# 初期化
u = np.empty((M+2, N+1))
# 境界条件を満たす関数
g = 1 - x/L
# 初期条件
u[:, 0] = 1 * (x < L/10) + (1 + np.cos((x-L/10)/(L/8)*np.pi))/2 * (np.logical_and(x >= L/10, x < L/10 + L/8)) 
# 斉次ディリクレ境界条件を満たす関数
v = u[:, 0] - g                        

# 初期条件のプロット
fig, ax = plt.subplots()
line, = ax.plot(x, u[:, 0], lw=2)
# アニメーションにおける update を定義 
def update(frame):
    line.set_ydata(u[:, frame])
    return line,

# 計算
for n in range(N): 
    f = r * u[:, n] * (1 - u[:, n])
    # v_hat = dst(v[1:-1], type=1, norm='ortho')
    # f_hat = dst(f[1:-1], type=1, norm='ortho')
    v_hat = 2/(M+1) * S @ v[1:-1]
    f_hat = 2/(M+1) * S @ f[1:-1]
    v_hat = (v_hat + tau * f_hat) / (1 + tau * k * (m*np.pi/L)**2)
    # v[1:-1] = idst1(v_hat, type=1, norm='ortho')
    v[1:-1] = S @ v_hat
    u[:, n+1] = v + g

# プロット
step: int = 2
ani = FuncAnimation(fig, update, frames=range(0, N, step), interval=5)

plt.show()
