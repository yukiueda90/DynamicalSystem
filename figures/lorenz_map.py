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
t = np.linspace(0, 100, 30000)  # 時間配列

# パラメータ
sigma = 10.0
beta = 8.0 / 3.0
rho = 28.0

# プロットの準備
fig = plt.figure(figsize=(10,4))
ax0 = fig.add_subplot(121, projection='3d')
ax1 = fig.add_subplot(122)
ax1.set_aspect('equal')

# odeintの設定と実行
solution = odeint(
    func=lorenz,
    y0=y0,
    t=t,
    args=(sigma, beta, rho),  # 関数への追加引数
    Dfun=jacobian,           # ヤコビ行列を指定
)
ax0.plot(solution[:, 0], solution[:, 1], solution[:, 2], color='tab:blue')
points = []
test = np.diff(solution[:,2])
for i, z in enumerate(solution[1:-1, 2]): 
    if test[i] > 0 and test[i+1] < 0: 
        points.append([solution[i, 0], solution[i, 1], z]) 
points = np.array(points)
t = np.linspace(0, 1, points.shape[0])
# ax.plot(points[:, 0], points[:, 1], points[:, 2], linestyle='None', marker='o', color='tab:orange')
ax0.scatter(points[:, 0], points[:, 1], points[:, 2], s=10, color='tab:orange')
ax0.set_xlabel("x")
ax0.set_ylabel("y")
ax0.set_zlabel("z")

ax1.plot([28, 48], [28, 48], linewidth=2, color='tab:orange')
ax1.scatter(points[:-1, 2], points[1:, 2], s=10, color='tab:blue')

fig.tight_layout()
plt.show()