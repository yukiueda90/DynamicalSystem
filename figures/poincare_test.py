import numpy as np

def lorenz(x, sigma=10, rho=28, beta=8/3):
    dx = sigma * (x[1] - x[0])
    dy = x[0] * (rho - x[2]) - x[1]
    dz = x[0] * x[1] - beta * x[2]
    return np.array([dx, dy, dz])

dt = 0.01
x = np.array([1.0, 1.0, 1.0])

z0 = 27
points = []

for _ in range(200000):
    x_prev = x.copy()
    
    # RK4
    k1 = lorenz(x)
    k2 = lorenz(x + dt*k1/2)
    k3 = lorenz(x + dt*k2/2)
    k4 = lorenz(x + dt*k3)
    x = x + dt*(k1 + 2*k2 + 2*k3 + k4)/6
    
    # 交差検出
    if x_prev[2] <= z0 and x[2] > z0:
        t = (z0 - x_prev[2]) / (x[2] - x_prev[2])
        x_cross = x_prev[0] + t*(x[0] - x_prev[0])
        points.append(x_cross)

points = np.array(points)
points = points[100:]

# 写像プロット
import matplotlib.pyplot as plt
plt.scatter(points[:-1], points[1:], s=1)
plt.show()