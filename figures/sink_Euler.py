import numpy as np 
import matplotlib.pyplot as plt

# logistic equation 
# x' = -x 
# y' = -3y

# INPUT 
# initial value 
x0 = np.array([1.0 , 1.0])
# time step size 
tau0: float = 0.5
tau1: float = 0.3
# number of time steps 
N: int = 10  
# right hand side 
rhs = lambda x: np.array([-x[0], -3*x[1]])
# exact solution 
exact = lambda t, x0: np.array([np.exp(-t) * x0[0], np.exp(-3*t) * x0[1]])

# solve ODE system 
def forward_Euler(x0, tau):
    # initialize array for result 
    result: np.ndarray = np.empty([2, N+1]) 
    result[:, 0] = x0
    for i in range(N):
        result[:, i+1] = result[:, i] + tau * rhs(result[:, i])
    return result 

t = np.linspace(0, N*tau0, 100)
exact_sol = exact(t, x0)
result0 = forward_Euler(x0, tau0)
result1 = forward_Euler(x0, tau1)
# plot 
fig, ax =plt.subplots() 
ax.plot(exact_sol[0], exact_sol[1], color='blue', label='exact solution')
ax.plot(result0[0], result0[1], color='red', marker='o', label='tau = 0.5')
ax.plot(result1[0], result1[1], color='green', marker='o', label='tau = 0.3')

ax.legend()
plt.savefig('sink_Euler.png')
plt.show()