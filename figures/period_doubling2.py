import numpy as np 
import matplotlib.pyplot as plt 

fig, ax = plt.subplots(figsize=(6, 4)) 
# for r in np.linspace(-0.1, 1.4, 1000): 
#     f = lambda x: -x - r*x + x**3 
for r in np.linspace(0.1, 4.0, 1000): 
    f = lambda x: r * x * (1-x)
    for x in np.linspace(0.1, 0.9, 5): 
        x_list = [x]
        for _ in range(10000): 
            x = f(x)
            x_list.append(x)
        ax.plot(r*np.ones(100), x_list[-101:-1], linestyle='None', marker='o', markersize=1, color='tab:blue')
plt.tight_layout()
plt.savefig('period_doubling_bifurcation.png')
plt.show()