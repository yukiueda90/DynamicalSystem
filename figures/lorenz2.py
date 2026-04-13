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

# rho > 1 で現れる平衡点 
x1 = lambda beta, rho: np.array([np.sqrt(beta*(rho-1)), np.sqrt(beta*(rho-1)), rho-1])
x2 = lambda beta, rho: np.array([-np.sqrt(beta*(rho-1)), -np.sqrt(beta*(rho-1)), rho-1])

# 初期条件と時間配列
y0 = [1.0, 1.0, 0.0]  # 初期値
t = np.linspace(0, 50, 100000)  # 時間配列

# パラメータ
sigma = 10.0
beta = 8.0 / 3.0

# プロットの準備
fig = plt.figure(figsize=(9, 4))
ax0 = fig.add_subplot(321, projection='3d')
ax1 = fig.add_subplot(322, projection='3d')
ax2 = fig.add_subplot(323, projection='3d')
ax3 = fig.add_subplot(324, projection='3d')
ax4 = fig.add_subplot(325, projection='3d')
ax5 = fig.add_subplot(326, projection='3d')
ax = [ax0, ax1, ax2, ax3, ax4, ax5]

# パラメータ
rho_list = [0.9, 5.0, 13, 13.926, 23.0, 24.8]

ax[0].plot(0, 0, 0, linestyle='None', marker='o', color='tab:green')
ax[1].plot(0, 0, 0, linestyle='None', marker='o', color='tab:red')
ax[1].plot(*x1(beta, rho_list[1]), linestyle='None', marker='o', color='tab:green')
ax[1].plot(*x2(beta, rho_list[1]), linestyle='None', marker='o', color='tab:green')
ax[2].plot(0, 0, 0, linestyle='None', marker='o', color='tab:red')
ax[2].plot(*x1(beta, rho_list[2]), linestyle='None', marker='o', color='tab:green')
ax[2].plot(*x2(beta, rho_list[2]), linestyle='None', marker='o', color='tab:green')
ax[3].plot(0, 0, 0, linestyle='None', marker='o', color='tab:red')
ax[3].plot(*x1(beta, rho_list[3]), linestyle='None', marker='o', color='tab:green')
ax[3].plot(*x2(beta, rho_list[3]), linestyle='None', marker='o', color='tab:green')
ax[4].plot(0, 0, 0, linestyle='None', marker='o', color='tab:red')
ax[4].plot(*x1(beta, rho_list[4]), linestyle='None', marker='o', color='tab:green')
ax[4].plot(*x2(beta, rho_list[4]), linestyle='None', marker='o', color='tab:green')
ax[5].plot(0, 0, 0, linestyle='None', marker='o', color='tab:red')
ax[5].plot(*x1(beta, rho_list[5]), linestyle='None', marker='o', color='tab:red')
ax[5].plot(*x2(beta, rho_list[5]), linestyle='None', marker='o', color='tab:red')


for i, rho in enumerate(rho_list):
    # odeintの設定と実行
    solution = odeint(
        func=lorenz,
        y0=y0,
        t=t,
        args=(sigma, beta, rho),  # 関数への追加引数
        Dfun=jacobian,           # ヤコビ行列を指定
    )
    ax[i].plot(solution[:, 0], solution[:, 1], solution[:, 2])
    ax[i].set_xlabel("x")
    ax[i].set_ylabel("y")
    ax[i].set_zlabel("z")
    ax[i].set_title(f'rho = {rho}')

fig.tight_layout()
plt.show()