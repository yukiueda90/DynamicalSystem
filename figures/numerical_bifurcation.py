import numpy as np
import matplotlib.pyplot as plt

# 関数の定義 
f = lambda x, r: r*x + x**3 - x**5 
fx = lambda x, r: r + 3*x**2 - 5*x**4 
fr = lambda x, r: x

# 疑似弧長法 
def pseudo_arclength(y0: np.ndarray, ds: float = 0.01, n_steps: int = 1000) -> np.ndarray: 
  # 配列の初期化
  y = np.empty((2, n_steps+1)) 
  y[:] = np.nan 
  y[:, 0] = y0 
  # 疑似弧長法の反復
  for i in range(n_steps): 
    v = np.array([fr(y[0, i], y[1, i]), -fx(y[0, i], y[1, i])]) 
    v = v / np.linalg.norm(v) 
    y_pred = y[:, i] + ds * v # 予測
    y[:, i+1] = newton_corrector(y_pred, v) # ニュートン法で修正

  return y

# 疑似弧長法におけるニュートン法
def newton_corrector(y_pred: np.ndarray, v: np.ndarray, tol: float = 1e-8, max_iter: int = 30): 
  # ニュートン法の初期値
#   y = y_pred.copy()
  y = y_pred
  # ニュートン法の反復
  for _ in range(max_iter):
    # ベクトル
    F = np.array([
        f(y[0], y[1]), 
        (y - y_pred) @ v
    ])
    # ヤコビ行列
    DF = np.array([
        [fx(y[0], y[1]), fr(y[0], y[1])],
        [v[0], v[1]]
    ])
    delta = np.linalg.solve(DF, F)
    y = y - delta # ニュートン法で更新
    # 収束したら計算終了
    if np.linalg.norm(delta) < tol:
        return y
  raise ValueError("Newton method did not converge.")

# プロットの準備 
fig, ax = plt.subplots() 
# 初期値を設定 
y0 = [0.01, np.nan] 
y0[1] = -(y0[0]**2 - y0[0]**4) 
# 疑似弧長法で計算 
y = pseudo_arclength(y0)
# 安定性の判定 
fx_value = fx(y[0], y[1])
stable = fx_value < 0
unstable = fx_value > 0
# プロット 
ax.plot(y[1, stable], y[0, stable], color='tab:blue', linewidth=2.5)
ax.plot(y[1, unstable], y[0, unstable], color='tab:blue', linewidth=2.5, linestyle='dashed')

# # 初期値を設定 
# y0 = [-0.01, np.nan] 
# y0[1] = -(y0[0]**2 - y0[0]**4) 
# # 疑似弧長法で計算 
# y = pseudo_arclength(y0)
# # 安定性の判定 
# fx_value = fx(y[0], y[1])
# stable = fx_value < 0
# unstable = fx_value > 0
# # プロット 
# ax.plot(y[1, stable], y[0, stable], color='tab:blue')
# ax.plot(y[1, unstable], y[0, unstable], color='tab:red')

# # 初期値を設定 
# y0 = [0., -1.0] 
# # 疑似弧長法で計算 
# y = pseudo_arclength(y0)
# # 安定性の判定 
# fx_value = fx(y[0], y[1])
# stable = fx_value < 0
# unstable = fx_value > 0
# # プロット 
# ax.plot(y[1, stable], y[0, stable], color='tab:blue')
# ax.plot(y[1, unstable], y[0, unstable], color='tab:red')

# # 初期値を設定 
# y0 = [0., 8.0] 
# # 疑似弧長法で計算 
# y = pseudo_arclength(y0)
# # 安定性の判定 
# fx_value = fx(y[0], y[1])
# stable = fx_value < 0
# unstable = fx_value > 0
# # プロット 
# ax.plot(y[1, stable], y[0, stable], color='tab:blue')
# ax.plot(y[1, unstable], y[0, unstable], color='tab:red')

plt.show()
