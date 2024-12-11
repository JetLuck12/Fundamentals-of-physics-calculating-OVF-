import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
from scipy.integrate import quad

def J_0(x):
    integrand = lambda t: np.cos(x * np.sin(t))
    integral, _ = quad(integrand, 0, np.pi)
    return integral / np.pi

def lagrange_interpolation(x_vals, y_vals):
    return lagrange(x_vals, y_vals)

def interpolate_and_plot(n_vals, x_range):
    x_full = np.linspace(x_range[0], x_range[1], 1000)
    y_exact = np.array([J_0(x) for x in x_full])

    plt.figure(figsize=(10, 6))

    for n in n_vals:
        # Узлы интерполяции
        x_nodes = np.linspace(x_range[0], x_range[1], int(n))
        y_nodes = np.array([J_0(x) for x in x_nodes])

        poly = lagrange_interpolation(x_nodes, y_nodes)
        y_interp = poly(x_full)

        # График разности P_n(x) - J_0(x)
        plt.plot(x_full, y_interp - y_exact, label=f'n={n}')

    plt.xlabel("x")
    plt.ylabel("$P_n(x) - J_0(x)$")
    plt.title("Error between Interpolation $P_n(x)$ and Exact $J_0(x)$")
    plt.grid(True)
    plt.legend()
    plt.show()

# Повторный запуск интерполяции и построения графиков
n_values = np.linspace(1,21,5)
interpolate_and_plot(n_values, [0, 10])
