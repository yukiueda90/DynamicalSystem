import numpy as np 
import matplotlib.pyplot as plt 

fig, ax = plt.subplots(1, 2) 
fig.tight_layout()
for i, r in enumerate([-0.2, 0.2]): 
    f = lambda x: x + r + x**2
    ax[i].set_aspect('equal')
    ax[i].set_xlim([-1.1, 1.1])
    ax[i].set_ylim([-1.1, 2.4])
    pts = np.linspace(-1.0, 1.0, 500) 
    ax[i].plot(pts, pts, linewidth=2, color='tab:orange')
    ax[i].plot(pts, f(pts), linewidth=2, color='tab:blue')
    if i == 0:
        x = 0.3 
        for _ in range(6): 
            ax[i].plot([x, x], [x, f(x)], color='k') 
            ax[i].arrow(x=x, y=x, dx=0, dy=(f(x)-x)/3, head_width=0.015, color='k')
            ax[i].plot([x, f(x)], [f(x), f(x)], color='k') 
            ax[i].arrow(x=x, y=f(x), dx=(f(x)-x)/3, dy=0, head_width=0.015, color='k')
            x = f(x)
plt.savefig('discrete_saddle_node.png')
plt.show()