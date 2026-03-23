import numpy as np
import matplotlib.pyplot as plt

# 調和振動子:
# x' = y, y' = -x
# 初期条件:
# x(0) = 1, y(0) = 0
# 厳密解: x = \cos(t), y = -sin(t)

# 行列の定義
A = np.array([[0, 1], [-1, 0]])

fig, ax = plt.subplots()
ax.set_aspect('equal')

# 時間区間分割
N: int = 150
t = np.linspace(0, 2*np.pi, N+1)
tau: float = 2*np.pi / N

# 厳密解のプロット
x = np.cos(t)
y = np.sin(t)
ax.plot(x, y, label='exact solution')

# 前進オイラー法
# 配列の初期化
x = np.empty((2, N+1))
# 初期値の代入
x[0, 0] = 1
x[1, 0] = 0
# 各ステップの計算
M = np.eye(2) + tau * A
for i in range(N):
  x[:, i+1] = M @ x[:, i]
# 数値解のプロット
ax.plot(x[0], x[1], label='forward Euler')

# 後退オイラー法
# 配列の初期化
x = np.empty((2, N+1))
# 初期条件の代入
x[0, 0] = 1
x[1, 0] = 0
# 各ステップの計算
M = np.eye(2) - tau * A
for i in range(N):
  x[:, i+1] = np.linalg.solve(M, x[:, i])
# 数値解のプロット
ax.plot(x[0], x[1], label='backward Euler')

# プロットの表示
ax.legend()
plt.show()
