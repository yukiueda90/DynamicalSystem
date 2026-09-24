import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from scipy.integrate import solve_ivp
from scipy.optimize import root

# ダフィング方程式
#   x' = y
#   y' = x - x^3 + gamma * cos(omega*t)

# パラメータ
gamma = 0.01
omega = 1.0

T = 2.0 * np.pi / omega

# 右辺のベクトル
def f(t: float, x: np.ndarray):
    return np.array([
        x[1],
        x[0] - x[0]**3 + gamma * np.cos(omega*t)
    ])

# # ヤコビ行列
# def Df(x): 
#     s = x[0]**2 + x[1]**2 
#     r1 = a * (s - 1) * (4 - s) 
#     r2 = a * (5 - 2*s)
#     return np.array([ 
#         [r1 + 2 * x[0]**2 * r2, -1 + 2 * x[0] * x[1] * r2], 
#         [1 + 2 * x[0] * x[1] * r2, r1 + 2 * x[1]**2 * r2]
#     ])

# 古典的ルンゲ=クッタ (４段陽的ルンゲ=クッタ)
def runge_kutta(t: float, x: np.ndarray, tau: float):
    k1 = f(t, x)
    k2 = f(t + tau/2, x + tau/2 * k1)
    k3 = f(t + tau/2, x + tau/2 * k2)
    k4 = f(t + tau, x + tau * k3)
    x = x + tau/6 * (k1 + 2*k2 + 2*k3 + k4) 
    return x

# プロット作成用にルンゲ=クッタ法を解く
def save_orbit(x0: np.ndarray, tau: float, N: int): 
    t = 0
    x = np.empty((2, N+1))
    x[:, 0] = x0
    for i in range(N):
        x[:, i+1] = runge_kutta(t, x[:, i], tau) 
        t += tau
    return x

# プロットの準備
fig, ax = plt.subplots() 
fig.tight_layout()
ax.set_aspect('equal')
point, = ax.plot([], [], 'o', color='tab:blue')
line, = ax.plot([], [], color='tab:blue')

# プロット用パラメータ
tau: float = 1e-2 
N: int = 10000
# 適当な初期値から軌道をプロット
x0 = np.array([
    0.27, 
    0.20
])

x = save_orbit(x0, tau, N) 
# ax.plot(x[0], x[1], color='tab:blue')

# plt.savefig('duffing_eq2.png')    
# plt.show()
# optional: fix axis limits (important!)
ax.set_xlim(-2.1, 2.1)
ax.set_ylim(-1.1, 1.1)

def update(frame):
    xn = x[:, frame]
    # wrap scalars as sequences
    point.set_data([xn[0]], [xn[1]])
    line.set_data(x[0, :frame+1], x[1, :frame+1])
    return point, line

step: int = 5
ani = FuncAnimation(fig, update, frames=range(0, x.shape[1], step), interval=10)
# ani.save("duffing_equation2.mp4", writer="ffmpeg", fps=60)

plt.show()

# ============================================================
# Poincare map P
#
# Integrate one forcing period T.
# ============================================================

# def poincare(z0):
#     sol = solve_ivp(
#         f,
#         [0.0, T],
#         z0,
#         method="DOP853",
#         rtol=1.0e-10,
#         atol=1.0e-12
#     )

#     return sol.y[:, -1]


# inverse Poincare map
#
# Because the forcing is T-periodic, integrating from 0 to -T
# gives the inverse stroboscopic map.
# def poincare_inverse(z0):
#     sol = solve_ivp(
#         duffing,
#         [0.0, -T],
#         z0,
#         method="DOP853",
#         rtol=1.0e-10,
#         atol=1.0e-12
#     )

#     return sol.y[:, -1]


# ============================================================
# Find a saddle fixed point:
#
#       P(p) = p
# ============================================================

# def fixed_point_equation(z):
#     return poincare(z) - z


# res = root(fixed_point_equation, [0.0, 0.0])

# if not res.success:
#     raise RuntimeError("Fixed point search failed.")

# p = res.x

# print("saddle fixed point")
# print(p)
# print()

# print("|P(p)-p| =",
#       np.linalg.norm(poincare(p) - p))
# print()


# # ============================================================
# # Jacobian DP(p) by finite differences
# # ============================================================

# def jacobian_poincare(z, h=1.0e-6):

#     J = np.zeros((2, 2))

#     for j in range(2):

#         e = np.zeros(2)
#         e[j] = 1.0

#         J[:, j] = (
#             poincare(z + h*e)
#             - poincare(z - h*e)
#         ) / (2.0*h)

#     return J


# J = jacobian_poincare(p)

# eigval, eigvec = np.linalg.eig(J)

# print("DP(p) =")
# print(J)
# print()

# print("eigenvalues =")
# print(eigval)
# print()


# # stable / unstable directions

# i_s = np.argmin(np.abs(eigval))
# i_u = np.argmax(np.abs(eigval))

# lambda_s = eigval[i_s]
# lambda_u = eigval[i_u]

# v_s = np.real(eigvec[:, i_s])
# v_u = np.real(eigvec[:, i_u])

# v_s /= np.linalg.norm(v_s)
# v_u /= np.linalg.norm(v_u)

# print("lambda_s =", lambda_s)
# print("lambda_u =", lambda_u)
# print()

# print("stable direction   =", v_s)
# print("unstable direction =", v_u)


# # ============================================================
# # Batch integration
# #
# # Used for moving many points simultaneously.
# # ============================================================

# def poincare_batch(Z, inverse=False):
#     """
#     Z.shape = (..., 2)

#     Return P(Z), or P^{-1}(Z).
#     """

#     shape = Z.shape

#     z0 = np.asarray(Z).reshape(-1, 2).ravel()

#     def rhs_batch(t, z):

#         dz = np.empty_like(z)

#         x = z[0::2]
#         y = z[1::2]

#         dz[0::2] = y

#         dz[1::2] = (
#             x
#             - x**3
#             - delta*y
#             + gamma*np.cos(omega*t)
#         )

#         return dz

#     tf = -T if inverse else T

#     sol = solve_ivp(
#         rhs_batch,
#         [0.0, tf],
#         z0,
#         method="DOP853",
#         rtol=1.0e-9,
#         atol=1.0e-11
#     )

#     return sol.y[:, -1].reshape(shape)


# # ============================================================
# # 1. Stable and unstable manifolds
# # ============================================================

# eps = 1.0e-5
# num_points = 301

# s = np.linspace(-eps, eps, num_points)

# # tiny line segment in the unstable eigendirection
# Wu0 = p[None, :] + s[:, None] * v_u[None, :]

# # tiny line segment in the stable eigendirection
# Ws0 = p[None, :] + s[:, None] * v_s[None, :]


# # forward iteration -> unstable manifold
# Wu = [Wu0]

# for k in range(9):
#     Wu.append(
#         poincare_batch(Wu[-1])
#     )


# # backward iteration -> stable manifold
# Ws = [Ws0]

# for k in range(6):
#     Ws.append(
#         poincare_batch(Ws[-1], inverse=True)
#     )


# fig, ax = plt.subplots(figsize=(8, 6))


# # unstable manifold
# for k, curve in enumerate(Wu):

#     mask = np.linalg.norm(curve, axis=1) < 3.0

#     ax.plot(
#         curve[mask, 0],
#         curve[mask, 1],
#         linewidth=1.0,
#         label=r"$W^u$" if k == 0 else None
#     )


# # stable manifold
# for k, curve in enumerate(Ws):

#     mask = np.linalg.norm(curve, axis=1) < 3.0

#     ax.plot(
#         curve[mask, 0],
#         curve[mask, 1],
#         "--",
#         linewidth=1.0,
#         label=r"$W^s$" if k == 0 else None
#     )


# ax.plot(
#     p[0],
#     p[1],
#     "ko",
#     markersize=6,
#     label="saddle fixed point"
# )


# ax.set_xlim(-1.6, 1.6)
# ax.set_ylim(-1.2, 1.2)

# ax.set_xlabel(r"$x$")
# ax.set_ylabel(r"$y$")

# ax.set_aspect("equal")
# ax.legend()

# ax.set_title(
#     "Stable and unstable manifolds of the Poincare map"
# )

# plt.tight_layout()
# plt.show()


# # ============================================================
# # 2. Deformation of a small rectangle R
# #
# # The rectangle is aligned with the stable and unstable
# # eigendirections.
# # ============================================================

# a_u = 2.0e-3
# a_s = 2.0e-3

# Nu = 21
# Ns = 21

# u = np.linspace(-a_u, a_u, Nu)
# s = np.linspace(-a_s, a_s, Ns)

# U, S = np.meshgrid(
#     u,
#     s,
#     indexing="ij"
# )


# # R = p + u v_u + s v_s
# Z0 = (
#     p[None, None, :]
#     + U[:, :, None] * v_u[None, None, :]
#     + S[:, :, None] * v_s[None, None, :]
# )


# # Compute P^n(R)
# Nmax = 8

# Z = [Z0]

# for n in range(Nmax):
#     Z.append(
#         poincare_batch(Z[-1])
#     )


# # ============================================================
# # Plot selected iterates
# # ============================================================

# plot_steps = [0, 4, 6, 8]

# fig, axes = plt.subplots(
#     2, 2,
#     figsize=(10, 8)
# )

# for ax, n in zip(axes.flat, plot_steps):

#     Zn = Z[n]

#     # grid lines in the stable direction
#     for i in range(Nu):
#         ax.plot(
#             Zn[i, :, 0],
#             Zn[i, :, 1],
#             linewidth=0.7
#         )

#     # grid lines in the unstable direction
#     for j in range(Ns):
#         ax.plot(
#             Zn[:, j, 0],
#             Zn[:, j, 1],
#             linewidth=0.7
#         )

#     ax.plot(
#         p[0],
#         p[1],
#         "ko",
#         markersize=4
#     )

#     ax.set_xlim(-1.5, 1.5)
#     ax.set_ylim(-1.0, 1.0)

#     ax.set_aspect("equal")

#     ax.set_title(
#         rf"$P^{{{n}}}(R)$"
#     )

#     ax.set_xlabel(r"$x$")
#     ax.set_ylabel(r"$y$")


# plt.tight_layout()
# plt.show()