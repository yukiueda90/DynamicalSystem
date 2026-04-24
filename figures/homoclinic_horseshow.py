import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

eps = 0.2
omega = 1.0
T = 2 * np.pi / omega

def f(t, z):
    x, y = z
    dx = y
    dy = np.sin(x) + eps * np.cos(omega * t)
    return [dx, dy]

def poincare(z0, n_iter=300):
    t0 = 0
    z = z0
    pts = []

    for _ in range(n_iter):
        sol = solve_ivp(f, [t0, t0 + T], z, max_step=0.05)
        z = [sol.y[0, -1], sol.y[1, -1]]
        pts.append(z)

    return np.array(pts)

plt.figure(figsize=(6,6))

# initial points near separatrix (homoclinic loop)
for _ in range(10):
    z0 = [np.pi + 0.01*np.random.randn(), 0.01*np.random.randn()]
    pts = poincare(z0)
    plt.plot(pts[:,0], pts[:,1], '.', markersize=1)

plt.xlim(-4, 4)
plt.ylim(-3, 3)
plt.title("Homoclinic tangle (Poincaré section)")
plt.show()