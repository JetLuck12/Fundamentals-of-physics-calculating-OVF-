import numpy as np
import matplotlib.pyplot as plt
from Task3.trapezoid import trapezoid_method
from Task3.Simpson import simpson_method

def J_m(m, x, method, n = 5):
    integrand = lambda t: np.cos(m * t - x * np.sin(t))
    integral = method(integrand, 0, np.pi, n)
    return integral / np.pi

x = 1.7335308262508478
N = 10000
h = 2*np.pi/N

x_vals = [x*h for x in range(1,N)]
def find_worst_x():
    J_vals_trap = J_vals_simp = [0 for x in range(N + 1)]
    error_vals_trap = error_vals_simp = [0 for x in range(1, N)]

    J_vals_simp[0] = (J_m(0, 0, simpson_method))
    J_vals_trap[0] = (J_m(0, 0, trapezoid_method))
    J_vals_simp[1] = (J_m(0, h, simpson_method))
    J_vals_trap[1] = (J_m(0, h, trapezoid_method))
    J_vals_simp[2] = (J_m(0, 2*h, simpson_method))
    J_vals_trap[2] = (J_m(0, 2*h, trapezoid_method))

    for i in range(2, N-1):
        J_vals_simp[i+1] = (J_m(0, (i+1)*h, simpson_method))
        J_vals_trap[i+1] = (J_m(0, (i+1)*h, trapezoid_method))
        J_vals_simp[i+2] = (J_m(0, (i+2)*h, simpson_method))
        J_vals_trap[i+2] = (J_m(0, (i+2)*h, trapezoid_method))

        error = (J_vals_simp[i-2]/12 - 2*J_vals_simp[i-1]/3 + 2*J_vals_simp[i+1]/3- J_vals_simp[i+2]/12)/h + J_m(1, i*h, method=simpson_method)
        error_vals_simp[i-1] = (abs(error))
        error2 = (J_vals_trap[i-2]/12 - 2*J_vals_trap[i-1]/3 + 2*J_vals_trap[i+1]/3- J_vals_trap[i+2]/12)/h + J_m(1, i*h, method=trapezoid_method)
        error_vals_trap[i-1] = (abs(error2))


    plt.figure(figsize = (10,6))

    plt.subplot(1,2,1)
    #plt.yscale('log')
    plt.plot(x_vals, error_vals_simp)
    plt.xlabel("x")
    plt.ylabel("Error")
    plt.title("Simpson error")
    plt.grid(True)

    plt.subplot(1,2,2)
    #plt.yscale('log')
    plt.plot(x_vals, error_vals_trap)
    plt.xlabel("x")
    plt.ylabel("Error")
    plt.title("Trapezoid error")
    plt.grid(True)
    plt.show()

    for i in range(N - 1):
        print(x_vals[i], error_vals_simp[i])

    print(max(error_vals_simp))


find_worst_x()