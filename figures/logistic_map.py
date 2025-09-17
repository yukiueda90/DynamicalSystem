import numpy as np 
import matplotlib.pyplot as plt

# logistic map 
# x' = a(1-x)x 

# INPUT 
# parameters
# a = 0.8
# a = 1.2
# a = 2.8
a = 3.6
# number of time step 
N: int = 40
# initial value 
x = 0.5

# logistic equations 
def logistic(x):
    return a * (1-x) * x  

# solve ODE system 
def solve(x0):
    # initialize array for result 
    result: np.ndarray = np.empty(N) 
    result[0] = x0
    for i in range(N-1):
        result[i+1] = logistic(result[i]) 
    return result 

# computation 
fname: str = 'data_logistic_map.txt' 
with open(fname, 'w') as f:
    f.write('# numerical result for logistic equation: \n') 
    f.write(f'# dx/dt = {a} * (1-x) * x. \n')
    f.write(f'# initial condition: x0 = {x} \n')

fig, ax = plt.subplots() 
result = np.vstack((range(N), solve(x)))    
ax.plot(result[0, :], result[1, :], color='b', linestyle='None', marker='o') 
with open(fname, 'a') as f: 
    for data in result.T:
        f.write(f'{data[0]} {data[1]} \n')
    f.write('\n\n')

plt.show()



