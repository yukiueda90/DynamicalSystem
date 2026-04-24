import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def horseshoe_map(x, y):
    # Step 1: stretch and contract
    x_new = 2 * x
    y_new = 0.5 * y

    # Step 2: fold
    mask = x_new > 1
    x_new[mask] = 2 - x_new[mask]
    y_new[mask] = 1 - y_new[mask]

    return x_new, y_new


# initial grid of points
n = 2000
x = np.random.rand(n)
y = np.random.rand(n)
y = np.array(sorted(y))

# iterate and plot

colors = np.linspace(0, 1, n)

fig, ax = plt.subplots(figsize=(6, 6))
scat = ax.scatter(x, y, c=colors, s=2, cmap='plasma')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title("Horseshoe Map")

def update(frame):
    global x, y
    x, y = horseshoe_map(x, y)
    scat.set_offsets(np.c_[x, y])
    return scat,

ani = FuncAnimation(fig, update, frames=range(20), interval=200)

# # Save as mp4
ani.save("horseshoe.mp4", dpi=150, fps=3)

plt.show()