import numpy as np 
import matplotlib.pyplot as plt 

f = lambda x: 2*x * (x<1/2) + 2*(1-x) * (x>=1/2) 

fig, ax = plt.subplots(1, 2) 
ax[0].set_aspect('equal')
ax[1].set_aspect('equal')
pts = np.linspace(0, 1.0, 500) 
ax[0].plot(pts, pts, linewidth=2, color='tab:orange')
ax[0].plot(pts, f(pts), linewidth=2, color='tab:blue')
ax[1].plot(pts, pts, linewidth=2, color='tab:orange')
ax[1].plot(pts, f(pts), linewidth=2, color='tab:blue')


x = 0.063 
for i in range(40): 
    ax[1].plot([x, x], [x, f(x)], color='k') 
    ax[1].arrow(x=x, y=x, dx=0, dy=(f(x)-x)/3, head_width=0.015, color='k')
    ax[1].plot([x, f(x)], [f(x), f(x)], color='k') 
    ax[1].arrow(x=x, y=f(x), dx=(f(x)-x)/3, dy=0, head_width=0.015, color='k')
    x = f(x)

fig.tight_layout()
plt.show()