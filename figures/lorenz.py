import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ローレンツ方程式の定義
def lorenz(y, t, sigma, beta, rho):
    x, y, z = y  # 状態変数を分解して表式を簡潔にする
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# ヤコビ行列（ローレンツ方程式の偏微分）を定義
def jacobian(y, t, sigma, beta, rho):
    x, y, z = y
    return [
        [-sigma, sigma, 0],
        [rho - z, -1, -x],
        [y, x, -beta]
    ]

# 初期条件と時間配列
y0 = [1.0, 1.0, 1.0]  # 初期値
t = np.linspace(0, 50, 300000)  # 時間配列

# パラメータ
sigma = 10.0
beta = 8.0 / 3.0
rho = 28.0

# odeintの設定と実行
solution = odeint(
    func=lorenz,
    y0=y0,
    t=t,
    args=(sigma, beta, rho),  # 関数への追加引数
    Dfun=jacobian,           # ヤコビ行列を指定
)

fig = plt.figure(figsize=(9, 4))
ax0 = fig.add_subplot(121, projection='3d')
ax1 = fig.add_subplot(122, projection='3d')
ax0.plot(solution[:, 0], solution[:, 1], solution[:, 2])
# ax.set_title("Lorenz Attractor")
y0 = [-1.0, -1.0, -1.0]  # 初期値
# odeintの設定と実行
solution = odeint(
    func=lorenz,
    y0=y0,
    t=t,
    args=(sigma, beta, rho),  # 関数への追加引数
    Dfun=jacobian,           # ヤコビ行列を指定
)
ax1.plot(solution[:, 0], solution[:, 1], solution[:, 2])
ax0.set_xlabel("x")
ax0.set_ylabel("y")
ax0.set_zlabel("z")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("z")
fig.tight_layout()
plt.show()