import numpy as np 
import matplotlib.pyplot as plt

# logistic equation 
# x' = ax(1-x) 

# INPUT 
# parameters
a = 2.0
# initial value 
x0 = 0.1
# time step size 
tau: float = 1.0
# number of time steps 
N: int = 20
# discretized point 
tn = tau * np.arange(N+1)

# exact solution 
exact = lambda x: 1 / (1-(1-1/x0)*np.exp(-a*x))

# solve ODE system 
def forward_Euler(x0):
    # initialize array for result 
    result: np.ndarray = np.empty(N+1) 
    result[0] = x0
    for i in range(N):
        result[i+1] = result[i] + a * tau * result[i]*(1 - result[i])
    return result 

def backward_Euler(x0): 
    # initialize array for result 
    result: np.ndarray = np.empty(N+1) 
    result[0] = x0
    for i in range(N):
        result[i+1] = (a*tau - 1 + np.sqrt((1-a*tau)**2 + 4*a*tau*result[i])) / (2*a*tau)
    return result 

fig, ax =plt.subplots(figsize=(6,4)) 
pts = np.linspace(0, N*tau, 300) 
ax.plot(pts, exact(pts), color='blue', label='exact solution')
ax.plot(tn, forward_Euler(x0), color='red', label='forward Euler') 
ax.plot(tn, backward_Euler(x0), color='green', label='backward Euler') 
ax.legend() 

plt.savefig('logistic_eq_Euler.png')
plt.show()

