import numpy as np 
import matplotlib.pyplot as plt 

# logistic map 
# x' = ax(1-x) 

# INPUT 
# number of time step 
N: int = 20
# list of parameter 
a_list = [1.2, 2.8, 3.6]
# plot
fig, ax = plt.subplots(1, 3, figsize=(10, 3)) 
pts = np.linspace(0, 1.0, 500) 

for i, a in enumerate(a_list):
    ax[i].set_aspect('equal')
    ax[i].plot(pts, pts, linewidth=2, color='tab:orange')
    f = lambda x: a * x * (1-x)
    ax[i].plot(pts, f(pts), linewidth=2, color='tab:blue') 
    # initial value 
    x = 0.5
    for _ in range(N): 
        ax[i].plot([x, x], [x, f(x)], color='k') 
        ax[i].arrow(x=x, y=x, dx=0, dy=(f(x)-x)/3, head_width=0.015, color='k')
        ax[i].plot([x, f(x)], [f(x), f(x)], color='k') 
        ax[i].arrow(x=x, y=f(x), dx=(f(x)-x)/3, dy=0, head_width=0.015, color='k')
        x = f(x)

# plt.savefig('logistic_cobweb.png')
plt.show()


