import numpy as np 
import matplotlib.pyplot as plt

# logistic equations 
# x' = (1-x)x - h 

# INPUT 
# parameters
# h = 0.16
# h = 0.25
h = 0.26
# computation time 
T: float = 1.9 
# number of time step 
N: int = 300
# initial value 
x = 1

# logistic equations 
def logistic(x):
    return (1-x) * x - h  

# Runge-Kutta method 
def runge_kutta(x, func, tau):
    """
    proceed one time step by explicit Runge-Kutta method 
    Args: 
        x: list[float], values at present time step 
        func: Callable, function f s.t. x' = f(x) 
        tau: float, time step size
    """
    k1 = func(x)
    k2 = func(x + tau/2*k1) 
    k3 = func(x + tau/2*k2) 
    k4 = func(x + tau*k3) 
    return x + tau/6 * (k1 + 2*k2 + 2*k3 + k4)

# solve ODE system 
def solve(x0):
    # initialize array for result 
    result: np.ndarray = np.empty(N) 
    result[0] = x0
    # time step size 
    tau: float = T/N 
    for i in range(N-1):
        result[i+1] = runge_kutta(result[i], logistic, tau) 
    return result 

# computation 
fname: str = 'data_logistic.txt' 
with open(fname, 'w') as f:
    f.write('# numerical result for logistic equation: \n') 
    f.write(f'# dx/dt = (1-x) * x - {h}. \n')

fig, ax = plt.subplots() 
t_list = np.linspace(0, T, N)
for i, x0 in enumerate(np.linspace(0.1, 1.1, 14)):
    result = np.vstack((t_list, solve(x0)))    
    ax.plot(result[0, :], result[1, :], color='b') 
    with open(fname, 'a') as f: 
        f.write(f'# index {i}, initial condition: x = {x0}: \n')
        for data in result.T:
            f.write(f'{data[0]} {data[1]} \n')
        f.write('\n\n')

plt.show()



