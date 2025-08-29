import numpy as np 
import matplotlib.pyplot as plt

# Lotka-Volterra equations 
# x' = ax - bxy 
# y' = cxy - dy 

# INPUT 
# parameters
a, b, c, d = 1., 1., 1., 1.
# computation time 
T: float = 9.0 
# number of time step 
N: int = 300
# initial value 
x, y = 1., 0.5 

# Rotka-Volterra equations 
def rotka_volterra(x):
    return np.array([a*x[0] - b*x[0]*x[1], c*x[0]*x[1] - d*x[1]])

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
def solve(x0, y0):
    # initialize array for result 
    result: np.ndarray = np.empty((2, N)) 
    result[:,0] = [x0, y0]
    # time step size 
    tau: float = T/N 
    for i in range(N-1):
        result[:, i+1] = runge_kutta(result[:, i], rotka_volterra, tau) 
    return result 

# computation 
fname: str = 'data_RotkaVolterra.txt' 
with open(fname, 'w') as f:
    f.write('# numerical result for Rotka-Volterra equations: \n') 
    f.write(f'# dx/dt = {a} * x - {b} * xy, \n')
    f.write(f'# dy/dt = {c} * xy - {d} * y \n')

fig, ax = plt.subplots() 
x0: float = 1.0
for i, y0 in enumerate(np.linspace(0.1, 0.8, 7)):
    result = solve(x0, y0)    
    ax.plot(result[0, :], result[1, :], color='b') 
    with open(fname, 'a') as f: 
        f.write(f'# index {i}, initial condition: (x, y) = ({x0}, {y0}): \n')
        for data in result.T:
            f.write(f'{data[0]} {data[1]} \n')
        f.write('\n\n')

plt.show()



