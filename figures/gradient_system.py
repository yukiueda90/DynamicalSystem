import numpy as np 
import matplotlib.pyplot as plt 


E = lambda x: ((x**3)/3 - (x**2)/2) 
dE = lambda x: (x**2 - x) 

# plot 
fig, ax = plt.subplots() 
x = np.linspace(-0.8, 1.8, 100) 
ax.plot(x, E(x)) 

# forward Euler 
tau = 3.0 
x = np.empty(5) 
x[:] = np.nan 
x[0] = 0.4 

for i in range(len(x)-1): 
    x[i+1] = x[i] - tau*dE(x[i]) 

ax.plot(x, E(x), linestyle='None', marker='o', color='tab:blue')
print(x)
plt.show()
