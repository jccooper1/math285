import matplotlib.pyplot as plt
import numpy as np
import os
def euler_method(f, y0, a, b, n):
    """
    Approximate solution of dy/dt = f(t, y) using Euler's method.
    :param f: The function f(t, y) defining the differential equation dy/dt = f(t, y).
    :param y0: The initial value of y at t=0.
    :param a: The starting value of the interval.
    :param b: The ending value of the interval.
    :param n: The number of steps.
    :return: Two lists, one of t values and one of y values.
    """
    os.environ["MKL_DEBUG_CPU_TYPE"] = "5"
    h = (b - a) / n  # Step size
    t_values = [a + i * h for i in range(n + 1)]
    y_values = [y0]  # Initial y value
    for i in range(n):
        t = t_values[i]
        y = y_values[i]
        y_new = y + h * f(t, y)
        y_values.append(y_new)
        
    return t_values, y_values
def f(t, y):
    return t + y
def exact_solution(t):
    return -t - 1 + np.exp(t)
# Set initial conditions and parameters
y0 = 1  # Initial y value
a = 0   # Start of the interval
b = 1   # End of the interval
n = 100  # Number of steps
# Get the values
t_values, y_values = euler_method(f, y0, a, b, n)
# Get the true values
true_values = [exact_solution(t) for t in t_values]
# Plot the true and estimated values
plt.figure(figsize=(10, 5))
plt.plot(t_values, y_values, 'o-', label='Estimated')
plt.plot(t_values, true_values, 'o-', label='True')
plt.title("Comparison of true and estimated values")
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
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
# Print results
for t, y, true, err in zip(t_values, y_values, true_values, error):
    print(f"t = {t:.2f}, y = {y:.4f}, true = {true:.4f}, error = {err:.4f}")