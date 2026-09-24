import numpy as np
import matplotlib.pyplot as plt

# x' = -y + x(x^2+y^2)
# y' = x + y(x^2+y^2)

# ベクトル場を定義
def f(x: np.ndarray) -> np.ndarray:
    return np.array([
        -x[1] + x[0] * (x[0]**2 + x[1]**2),
        x[0] + x[1] * (x[0]**2 + x[1]**2)
    ])
# ヤコビ行列を計算 
def df(x: np.ndarray) -> np.ndarray: 
    return np.array([
        [3*x[0]**2 + x[1]**2, -1 + 2*x[0]*x[1]], 
        [1 + 2*x[0]*x[1], x[0]**2 + 3*x[1]**2]
    ])
# 後退オイラー法: F(x_{n+1}) = 0 を満たす x_{n+1} を求める
def F(x: np.ndarray, x_prev: np.ndarray, tau: float) -> np.ndarray: 
    return x - tau*f(x) - x_prev 
def DF(x: np.ndarray, tau: float) -> np.ndarray: 
    return np.eye(2) - tau*df(x)

# ニュートン法: F(x) = 0 を解く
def newton(x_prev: np.ndarray, tau: float, tol: float = 1e-8, max_iter: int = 30) -> np.ndarray: 
    x = x_prev.copy()
    for _ in range(max_iter):
        residual = F(x, x_prev, tau)
        if np.linalg.norm(residual) < tol:
            return x
        # ニュートン法の反復: x_{k+1} = x_k - DF(x_k)^{-1}F(x_k)
        dx = np.linalg.solve(DF(x, tau), residual)
        x -= dx
    raise ValueError('Newton method did not converge.')

fig, ax = plt.subplots()
ax.set_aspect('equal')

# 時間刻み
tau: float = 3e-3
N: int = 10000

# 初期条件
x0: float = 0.1
y0: float = 0

# 配列を初期化
x = np.empty((2, N+1), dtype = float)
x[:, 0] = [x0, y0]

# ニュートン法
for i in range(N): 
   x[:, i+1] = newton(x[:, i], tau)

# 数値解のプロット
ax.plot(x[0], x[1])
# 向きを表す矢印のプロット 
idx = np.arange(100, N, 250) # 矢印の位置
vec = f(x[:, idx]) # 矢印の向き
norm = np.linalg.norm(vec, axis=0)
vec = vec / norm # 矢印の大きさを正規化
ax.quiver(x[0, idx], x[1, idx], vec[0], vec[1], scale=30, headwidth=3, minshaft=0, color='tab:blue')
# 初期値のプロット
ax.plot(x[0, 0], x[1, 0], linestyle='None', marker='o', label='initial value') 

ax.legend()
plt.savefig('linear_stability_analysis.png', bbox_inches='tight')
plt.show()