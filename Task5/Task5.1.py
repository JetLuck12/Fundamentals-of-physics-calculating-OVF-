import numpy as np
import matplotlib.pyplot as plt
from math import factorial

def get_chebyshev(a,b,N):
    cheb = lambda k: 1/2*(a+b) + 1/2*(b-a)*np.cos((2*k-1)*np.pi/(2*N))
    return np.array([ cheb(k) for k in range(1,N+1)])

def J0(x, terms=50):
    sum = 0
    for k in range(terms):
        term = ((-1) ** k * (x ** 2 / 4) ** k) / (factorial(k) ** 2)
        sum += term
    return sum


def lagrange_interpolation(x_values, y_values, x):
    n = len(x_values)
    result = 0
    for i in range(n):
        term = y_values[i]
        for j in range(n):
            if j != i:
                term *= (x - x_values[j]) / (x_values[i] - x_values[j])
        result += term
    return result

def plot_interpolation_difference(num_nodes_list, x_range=(0, 10), num_points=1000):
    x_plot = get_chebyshev(x_range[0], x_range[1], num_points)
    y_exact = [J0(x) for x in x_plot]

    plt.figure(figsize=(10, 6))


    i = 0
    for num_nodes in num_nodes_list:
        i += 1
        x_nodes = get_chebyshev(x_range[0], x_range[1], num_nodes)
        print(x_nodes)
        y_nodes = [J0(x) for x in x_nodes]

        y_interp = [lagrange_interpolation(x_nodes, y_nodes, x) for x in x_plot]

        y_diff = np.array(y_interp) - np.array(y_exact)
        plt.subplot(len(num_nodes_list)//4+1,4, i)
        plt.xlabel('x')
        plt.ylabel('P_n(x) - J_0(x)')
        plt.grid(True)
        plt.plot(x_plot, y_diff)

    plt.title('Графики разности P_n(x) - J_0(x)')
    plt.xlabel('x')
    plt.ylabel('P_n(x) - J_0(x)')
    plt.grid(True)
    plt.show()


num_nodes_list = [5,10,15,20]
plot_interpolation_difference(num_nodes_list)
