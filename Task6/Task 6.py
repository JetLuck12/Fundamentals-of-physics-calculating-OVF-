import numpy as np
import matplotlib.pyplot as plt

t0 = 0
t_end = 3
x0 = 1
N = 10
h = (t_end - t0) / N


def f(t, x):
    return -x


def euler_method(t0, x0, h, N):
    t_values = [t0]
    x_values = [x0]

    t = t0
    x = x0

    for _ in range(N):
        x = x + h * f(t, x)
        t += h
        t_values.append(t)
        x_values.append(x)

    return t_values, x_values


def runge_kutta_2nd_order(t0, x0, h, N):
    t_values = [t0]
    x_values = [x0]

    t = t0
    x = x0

    alpha = 3/4

    for _ in range(N):
        k1 = f(t, x)
        k2 = (1-alpha)*f(t,x) + alpha * f(t + h / (2*alpha) , x + h * k1 / (2*alpha))
        x = x + h * k2
        t += h
        t_values.append(t)
        x_values.append(x)

    return t_values, x_values


def runge_kutta_4th_order(t0, x0, h, N):
    t_values = [t0]
    x_values = [x0]

    t = t0
    x = x0

    for _ in range(N):
        k1 = f(t, x)
        k2 = f(t + h / 2, x + h * k1 / 2)
        k3 = f(t + h / 2, x + h * k2 / 2)
        k4 = f(t + h,     x + h * k3)
        x = x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t += h
        t_values.append(t)
        x_values.append(x)

    return t_values, x_values

def find_accuracy_order():

    node_list = [i for i in range(1, 100, 6)]

    euler_node_error = []
    rk2_node_error = []
    rk4_node_error = []

    for node in node_list:
        t_exact = np.linspace(t0, t_end, node + 1)
        x_exact = np.exp(-t_exact)

        # Решения численными методами
        t_euler, x_euler = euler_method(t0, x0, (t_end - t0) / node, node)
        t_rk2, x_rk2 = runge_kutta_2nd_order(t0, x0, (t_end - t0) / node, node)
        t_rk4, x_rk4 = runge_kutta_4th_order(t0, x0, (t_end - t0) / node, node)

        euler_errors = abs(np.array(x_euler) - np.array(x_exact))
        runge_2_errors = abs(np.array(x_rk2) - np.array(x_exact))
        runge_4_errors = abs(np.array(x_rk4) - np.array(x_exact))

        euler_node_error.append(max(euler_errors))
        rk2_node_error.append(max(runge_2_errors))
        rk4_node_error.append(max(runge_4_errors))
        print(euler_errors)

    # Построение графиков
    plt.figure(figsize=(12, 6))
    plt.yscale('log')
    plt.plot(node_list, euler_node_error, label="Euler method", marker="o")
    plt.plot(node_list, rk2_node_error, label="Runge-Kutta 2nd order", marker="x")
    plt.plot(node_list, rk4_node_error, label="Runge-Kutta 4th order", marker="s")
    plt.xlabel("n")
    plt.ylabel("error")
    plt.legend()
    plt.grid()
    plt.title("Numerical Solutions of dx/dt = -x")
    plt.show()




find_accuracy_order()

