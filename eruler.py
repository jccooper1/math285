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
    h = (b - a) / n  # Step size
    t_values = [a + i * h for i in range(n + 1)]
    y_values = [y0]  # Initial y value
    for i in range(n):
        t = t_values[i]
        y = y_values[i]
        y_new = y + h * f(t, y)
        y_values.append(y_new)
    return t_values, y_values
# Example usage:
# Define the function f(t, y). For instance, let's take f(t, y) = t + y.
def f(t, y):
    return t + y
# Set initial conditions and parameters
y0 = 0  # Initial y value
a = 0   # Start of the interval
b = 1   # End of the interval
n = 10  # Number of steps
# Get the values
t_values, y_values = euler_method(f, y0, a, b, n)
# Print results
for t, y in zip(t_values, y_values):
    print(f"t = {t:.2f}, y = {y:.4f}")
