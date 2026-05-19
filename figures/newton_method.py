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

# ニュートン法
def newton(x_prev: np.ndarray, tau: float, tol: float = 1e-8, max_iter: int = 30) -> np.ndarray: 
    x = x_prev 
    for _ in range(max_iter): 
        F = x - tau*f(x) - x_prev 
        DF = np.eye(2) - tau*df(x)
        x = x - np.linalg.solve(DF, F)
    if np.linalg.norm(np.linalg.solve(DF, F)) < tol:
       return x
    raise ValueError('Newton method did not converge.')

fig, ax = plt.subplots()
ax.set_aspect('equal')

# 時間区間分割
T: float = 30
N: int = 10000
t = np.linspace(0, T, N+1)
tau: float = T/N

# 初期条件
x0: float = 0.1
y0: float = 0

# 配列を初期化
x = np.empty((2, N+1), dtype = float)
x[:, 0] = [x0, y0]

# ニュートン法
for i in range(N): 
   x[:, i+1] = newton(x[:, i], tau)

# プロット
ax.plot(x[0], x[1])
# 向きを表す矢印 
vec = f(x)
ax.quiver(x[0][50:N:50], x[1][50:N:50], vec[0][50:N:50], vec[1][50:N:50], scale=5, headwidth=3, minshaft=0, color='tab:blue')
# 初期値
ax.plot(x[0, 0], x[1, 0], linestyle='None', marker='o', label='initial value') 
ax.legend()
plt.show()