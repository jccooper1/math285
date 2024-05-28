import numpy as np
import matplotlib.pyplot as plt

# Define the differential equations
def f1(t, y):
    return y**2 - t**2

def f2(t, y):
    return y**3 - t**2 * y

# Adams-Bashforth Predictor (4th order)
def adams_bashforth_predictor(y, t, h, f):
    f_values = [f(t[i], y[i]) for i in range(-4, 0)]
    return y[-1] + h * (55 * f_values[-1] - 59 * f_values[-2] + 37 * f_values[-3] - 9 * f_values[-4]) / 24

# Adams-Moulton Corrector (4th order)
def adams_moulton_corrector(y, t, h, f, y_pred):
    f_values = [f(t[i], y[i]) for i in range(-3, 0)]
    return y[-1] + h * (9 * f(t[-1] + h, y_pred) + 19 * f_values[-1] - 5 * f_values[-2] + f_values[-3]) / 24

# Milne-Simpson Method
def milne_simpson(y, t, h, f):
    f_values = [f(t[i], y[i]) for i in range(-3, 1)]
    y_pred = y[-4] + 4 * h * (2 * f_values[-1] - f_values[-2] + 2 * f_values[-3]) / 3
    return y[-2] + h * (f(t[-1] + h, y_pred) + 4 * f_values[-1] + f_values[-2]) / 3

# Hamming Method
def hamming(y, t, h, f):
    f_values = [f(t[i], y[i]) for i in range(-4, 0)]
    y_pred = y[-4] + 4 * h * (2 * f_values[-1] - f_values[-2] + 2 * f_values[-3]) / 3
    return (9 * y[-1] - y[-3] + 3 * h * (f(t[-1] + h, y_pred) + f_values[-1])) / 8

# General multistep solver
def multistep_solver(f, y0, t0, tf, h, method):
    t = np.arange(t0, tf + h, h)
    y = np.zeros(len(t))
    y[0] = y0
    
    # Initial conditions with Runge-Kutta of 4th order (RK4)
    for i in range(1, 4):
        k1 = h * f(t[i - 1], y[i - 1])
        k2 = h * f(t[i - 1] + h / 2, y[i - 1] + k1 / 2)
        k3 = h * f(t[i - 1] + h / 2, y[i - 1] + k2 / 2)
        k4 = h * f(t[i], y[i - 1] + k3)
        y[i] = y[i - 1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    
    # Apply multistep method
    for i in range(3, len(t) - 1):
        if method == 'Adams-Bashforth-Moulton':
            y_pred = adams_bashforth_predictor(y[i-3:i+1], t[i-3:i+1], h, f)
            y[i+1] = adams_moulton_corrector(y[i-2:i+1], t[i-2:i+1], h, f, y_pred)
        elif method == 'Milne-Simpson':
            y[i+1] = milne_simpson(y[i-3:i+1], t[i-3:i+1], h, f)
        elif method == 'Hamming':
            y[i+1] = hamming(y[i-3:i+1], t[i-3:i+1], h, f)
    
    return t, y

# Parameters
y0 = 1  # Initial condition y(0) = 1 for both equations
t0 = 0
tf = 0.5
h = 0.001

# Solve with different methods for both equations
methods = ['Adams-Bashforth-Moulton', 'Milne-Simpson', 'Hamming']
functions = [f1, f2]
solutions = {}

for func in functions:
    for method in methods:
        t, y = multistep_solver(func, y0, t0, tf, h, method)
        solutions[(func.__name__, method)] = (t, y)

# Plot results
plt.figure(figsize=(12, 8))
for func in functions:
    for method in methods:
        t, y = solutions[(func.__name__, method)]
        plt.plot(t, y, label=f'{func.__name__} - {method}')

plt.title('Comparison of Multistep Methods for Differential Equations with y(0)=1')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.grid(True)
plt.show()