import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

eps = 0.1

def f(t, z):
    x, y = z
    dx = y
    dy = x - x**3 + eps * y
    return [dx, dy]

# Saddle point near (1, 0)
saddle = [1, 0]

# Small perturbations along unstable direction
perturbations = np.linspace(-1, 1, 11)

plt.figure(figsize=(6,6))

# Unstable manifold (forward time)
for p in perturbations:
    z0 = [1 + p, 0.0]
    sol = solve_ivp(f, [0, 10], z0, max_step=0.01)
    plt.plot(sol.y[0], sol.y[1], 'r', alpha=0.6)

# Stable manifold (backward time)
for p in perturbations:
    z0 = [1 + p, 0.0]
    sol = solve_ivp(f, [0, -10], z0, max_step=0.01)
    plt.plot(sol.y[0], sol.y[1], 'b', alpha=0.6)

plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.title("Stable (blue) and Unstable (red) manifolds")
plt.show()