import numpy as np
from scipy.fft import dst, idst
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# parameters
L, N = 50.0, 1280
D, r = 1.0, 5.0
dt, steps = 0.003, 500

# grid (interior points)
x = np.linspace(0, L, N+2)[1:-1]

# boundary function g(x): g(0)=1, g(L)=0
g = 1 - x / L

# initial condition for u, then v = u - g
# u = g.copy()  
u = np.hstack((np.ones(80), (1 + np.cos(np.linspace(0, np.pi, 80)))/2, np.zeros(N-160)))                     # start near steady profile
v = u - g                         # satisfies homogeneous BCs

# wavenumbers
n = np.arange(1, N+1)
k2 = (n * np.pi / L)**2

# figure
fig, ax = plt.subplots()
line, = ax.plot(x, u, lw=2)
ax.set_ylim(-0.1, 1.1)
ax.set_title("Fisher-KPP evolution")
ax.set_xlabel("x")
ax.set_ylabel("u")

def step():
    global v
    u = v + g
    f = r * u * (1 - u)

    v_hat = dst(v, type=1, norm='ortho')
    f_hat = dst(f, type=1, norm='ortho')

    v_hat = (v_hat + dt * f_hat) / (1 + dt * D * k2)
    v = idst(v_hat, type=1, norm='ortho')

    return v + g

def update(frame):
    u_new = step()
    line.set_ydata(u_new)
    return line,

ani = FuncAnimation(fig, update, frames=100, interval=10)
plt.show()
