import numpy as np 
import matplotlib.pyplot as plt 

f = lambda x: 2*x * (x<1/2) + 2*(1-x) * (x>=1/2) 

fig, ax = plt.subplots(1, 2, figsize=(8, 4)) 
ax[0].set_aspect('equal')
ax[1].set_aspect('equal')
pts = np.linspace(0, 1.0, 500) 
ax[0].plot(pts, pts, linewidth=2, color='tab:orange')
ax[0].plot(pts, f(f(pts)), linewidth=2, color='tab:blue')
ax[1].plot(pts, pts, linewidth=2, color='tab:orange')
ax[1].plot(pts, f(f(f(pts))), linewidth=2, color='tab:blue')

fig.tight_layout()
plt.savefig('tent_map2.png')
plt.show()