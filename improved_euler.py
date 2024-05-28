import matplotlib.pyplot as plt
import numpy as np

def improved_euler_method(f, y0, a, b, n):
    # Initialize lists to store t and y values
    t_values = [a]
    y_values = [y0]
    # Calculate step size
    h = (b - a) / n
    # Improved Euler's method
    for _ in range(n):
        t_old = t_values[-1]
        y_old = y_values[-1]
        f_old = f(t_old, y_old)
        # Predictor step
        t_new = t_old + h
        y_new_predictor = y_old + h * f_old
        # Corrector step
        f_new = f(t_new, y_new_predictor)
        y_new_corrector = y_old + h * (f_old + f_new) / 2
        # Check if y is too large
        if abs(y_new_corrector) > 1e100:
            print("y value too large, stopping calculation")
            break
        # Append new values to the lists
        t_values.append(t_new)
        y_values.append(y_new_corrector)
    return t_values, y_values

def f(t, y):
    return y**2 - t**2

def exact_solution(t):
    return -t - 1 + np.exp(t)

y0 = 1
a = 0  # start of the interval
b = 0.6  # end of the interval
n = 100  # number of steps

t_values, y_values = improved_euler_method(f, y0, a, b, n)
# Get the true values
true_values = [exact_solution(t) for t in t_values]

# Calculate the error
error = [abs(y - true) for y, true in zip(y_values, true_values)]

# Plot the error
plt.figure(figsize=(10, 5))
plt.plot(t_values, error, 'o-', label='Error')
plt.title("Error of the approximation")
plt.xlabel('t')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.show()