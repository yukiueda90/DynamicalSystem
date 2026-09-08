import numpy as np 
import matplotlib.pyplot as plt

# logistic equation 
# x' = ax(1-x) 

# INPUT 
# parameters
a = 2.0
# initial value 
x0 = 0.1
# final time  
T: float = 2.0
# right hand side 
rhs = lambda x: a * x * (1-x)
# exact solution 
exact = lambda x: 1 / (1-(1-1/x0)*np.exp(-a*x))

# solve ODE system 
def forward_Euler(x0, tau):
    # initialize array for result 
    result: np.ndarray = np.empty(N+1) 
    result[0] = x0
    for i in range(N):
        result[i+1] = result[i] + a * tau * result[i]*(1 - result[i])
    return result 

def midpoint_method(x0, tau): 
    # initialize array for result 
    result: np.ndarray = np.empty(N+1) 
    result[0] = x0
    for i in range(N):
        k1 = rhs(result[i]) 
        k2 = rhs(result[i] + tau/2 * k1)
        result[i+1] = result[i] + tau * k2
    return result 

def RungeKutta(x0, tau): 
    # initialize array for result 
    result: np.ndarray = np.empty(N+1) 
    result[0] = x0
    for i in range(N):
        k1 = rhs(result[i]) 
        k2 = rhs(result[i] + tau/2 * k1)
        k3 = rhs(result[i] + tau/2 * k2) 
        k4 = rhs(result[i] + tau * k3)
        result[i+1] = result[i] + tau/6 * (k1 + 2*k2 + 2*k3 + k4)
    return result 

err_forwardEuler = [] 
err_midpoint = [] 
err_RungeKutta = []
tau_list = []

# number of time steps 
for N in [5, 10, 15, 20, 25, 30, 35, 40]:
    # time step size 
    tau: float = T/N 
    tau_list.append(tau)
    # forward Euler 
    result = forward_Euler(x0, tau) 
    error = abs(exact(T) - result[-1]) 
    err_forwardEuler.append(error) 
    # midpoint method 
    result = midpoint_method(x0, tau) 
    error = abs(exact(T) - result[-1]) 
    err_midpoint.append(error) 
    # Runge-Kutta 
    result = RungeKutta(x0, tau) 
    error = abs(exact(T) - result[-1]) 
    err_RungeKutta.append(error) 
        
# plot 
fig, ax =plt.subplots() 
ax.plot(tau_list, err_forwardEuler, color='red', marker='o', label='forward Euler')
ax.plot(tau_list, err_midpoint, color='green', marker='o', label='midpoint method') 
ax.plot(tau_list, err_RungeKutta, color='blue', marker='o', label='Runge-Kutta') 
ax.legend() 
# show slope
xs = 0.25 
ys = 5e-6 
ratio = 1.6 
x1 = xs * ratio 
y1 = ys * ratio**1 
ax.plot([xs, x1, x1, xs], [ys, ys, y1, ys], color='k') 
x2 = xs * ratio 
y2 = ys * ratio**2 
ax.plot([xs, x2, x2, xs], [ys, ys, y2, ys], color='k') 
x4 = xs * ratio 
y4 = ys * ratio**4 
ax.plot([xs, x4, x4, xs], [ys, ys, y4, ys], color='k') 


ax.set_xscale('log') 
ax.set_yscale('log')

plt.savefig('logistic_convergence.png')
plt.show()

