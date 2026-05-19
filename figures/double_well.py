import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# パラメータ 
alpha: float = 0.5  

# 空間領域, 時間区間
L: float = 1.5 
T: float = 10.0 
M: int = 300 
N: int = 1000 
x = np.linspace(-1.0, L, M+1) 
y = np.linspace(-1.0, L, M+1) 
t = np.linspace(0, T, N+1)
h: float = L/M 
tau: float = T/N 

# エネルギー 
E = lambda x: 1/4*(x[0]**2-1)**2 + 1/4*(x[1]**2-1)**2 + alpha/2*(x[0]-x[1])**2
dE = lambda x: np.array([ 
    x[0]*(x[0]**2-1) + alpha*(x[0]-x[1]), 
    x[1]*(x[1]**2-1) + alpha*(x[1]-x[0]) 
])

# エネルギーのプロット 
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
xx, yy = np.meshgrid(x, y)
ax.plot_surface(xx, yy, E([xx, yy]), cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.5)
plt.savefig('double_well.pdf')

# 初期化
x0 = [-0.85, 0.9]
x = np.empty((2, N+1))
x[:, 0] = x0 

# 初期条件のプロット
point, = ax.plot([], [], [], 'o', color='tab:blue')
line, = ax.plot([], [], [], color='tab:blue')

# optional: fix axis limits (important!)
# ax.set_xlim(-0.1, 1.1)
# ax.set_ylim(-0.1, 1.1)
# ax.set_zlim(-0.1, 1.1)
# ax.set_position([0, 0, 1, 1])

def update(frame):
    xn = x[:, frame]

    # wrap scalars as sequences
    point.set_data([xn[0]], [xn[1]])
    point.set_3d_properties([E(xn)])

    line.set_data(x[0, :frame+1], x[1, :frame+1])
    line.set_3d_properties(E(x[:, :frame+1]))

    return point, line


# 古典的ルンゲ=クッタ (４段陽的ルンゲ=クッタ)
def runge_kutta(x: np.ndarray, tau: float) -> np.ndarray:
    k1 = -dE(x)
    k2 = -dE(x + tau/2 * k1)
    k3 = -dE(x + tau/2 * k2)
    k4 = -dE(x + tau * k3)
    return x + tau/6 * (k1 + 2*k2 + 2*k3 + k4)

# 計算
for n in range(N): 
    x[:, n+1] = runge_kutta(x[:, n], tau)
    
# プロット
step: int = 3
ani = FuncAnimation(fig, update, frames=range(0, N, step), interval=3)
# ファイル出力する場合は下のコメントアウトを解除
ani.save("double_well.mp4", writer="ffmpeg", fps=60)

plt.show()
# Google Colab でアニメーションを表示するには 
# plt.show() をコメントアウトし, 代わりに以下のようにする (時間がかかるので注意): 
# from matplotlib import rc
# from IPython.display import HTML
# rc('animation', html='jshtml')
# ani 
