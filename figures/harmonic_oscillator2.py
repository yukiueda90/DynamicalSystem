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

# 時間区間分割の定義
T: float = 2*np.pi
N: int = 150
tau: float = T / N # 時間刻み幅
tn = tau * np.arange(N+1)

# 厳密解
x = np.cos(tn)
y = -np.sin(tn)
exact = np.linalg.norm(np.vstack((x, y)), axis=0) # ノルム
ax.plot(tn, exact, label='exact solution')

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
forward_euler = np.linalg.norm(x, axis=0) # ノルム
ax.plot(tn, forward_euler, label='Forward Euler')

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
backward_euler = np.linalg.norm(x, axis=0) # ノルム
ax.plot(tn, backward_euler, label='Backward Euler')

# 陰的中点法 
# 配列の初期化
x = np.empty((2, N+1))
# 初期条件の代入
x[0, 0] = 1
x[1, 0] = 0
# 各ステップの計算
M1 = np.eye(2) - tau/2 * A 
M2 = np.eye(2) + tau/2 * A
for i in range(N):
  x[:, i+1] = np.linalg.solve(M1, M2 @ x[:, i])
# 数値解のプロット
midpoint_method = np.linalg.norm(x, axis=0) # ノルム
ax.plot(tn, midpoint_method, label='Implicit Midpoint')

# シンプレクティックオイラー法 
# 配列の初期化
x = np.empty((2, N+1))
# 初期条件の代入
x[0, 0] = 1
x[1, 0] = 0
# 各ステップの計算
M1 = np.array([[1, 0], [-tau, 1]]) 
M2 = np.array([[1, tau], [0, 1]])
for i in range(N):
  x[:, i+1] = M1 @ M2 @ x[:, i]
# 数値解のプロット
symplectic_euler = np.linalg.norm(x, axis=0) # ノルム
ax.plot(tn, symplectic_euler, label='Symplectic Euler')

# プロットの出力
ax.legend()
plt.savefig('harmonic_oscillator2.png')
plt.show()
