import numpy as np 
import matplotlib.pyplot as plt 

f = lambda x: np.sqrt(x) 

fig, ax = plt.subplots() 
ax.set_aspect('equal')
pts = np.linspace(0, 1.2, 500) 
ax.plot(pts, pts, linewidth=2, color='tab:orange')
ax.plot(pts, f(pts), linewidth=2, color='tab:blue')

x = 0.1 
for i in range(6): 
    ax.plot([x, x], [x, f(x)], color='k') 
    ax.arrow(x=x, y=x, dx=0, dy=(f(x)-x)/3, head_width=0.015, color='k')
    ax.plot([x, f(x)], [f(x), f(x)], color='k') 
    ax.arrow(x=x, y=f(x), dx=(f(x)-x)/3, dy=0, head_width=0.015, color='k')
    x = f(x)

plt.show()