import numpy as np
import dyhotomy
import simple_iterations
from Task2.Newton import newton_method

U0 = 50
a = 1
tol = 1e-6
E_min = -U0
E_max = 0

def f(E_, u0_, a_):
    if -u0_ <= E_ < 0:
        k = np.sqrt(2 * (E_ + u0_))
        kappa = np.sqrt(2 * np.abs(E_))
        return k * np.tan(k * a_) - kappa
    else:
        return np.inf

def df(E, U0_, a_):
    if -U0_ <= E < 0:
        k = np.sqrt(2 * (E + U0_))
        d_k_d_e = 1 / np.sqrt(2 * (E + U0_))
        d_kappa_d_e = -1 / np.sqrt(2 * np.abs(E))
        return d_k_d_e * np.tan(k * a_) + k * a_ * (1 / np.cos(k * a_)) ** 2 * d_k_d_e - d_kappa_d_e
    else:
        return np.inf



print(f"for dichotomy method {dyhotomy.bisection_method(f, U0, a, E_min, E_max, tol)}")

max_iter = 10000
E_init_1 = -U0 / 2

try:
    ground_state_energy = simple_iterations.simple_iteration_method(U0, a, E_init_1, tol, max_iter)
    print(f"for iteration method: {ground_state_energy:.6f} (в произвольных единицах)")
except ValueError as e:
    print(e)

E_init_2 = -U0 / 2

try:
    ground_state_energy, iterations = newton_method(f, df, U0, a, E_init_2, tol, max_iter)
    print(f"for Newton method: {ground_state_energy:.6f} (в произвольных единицах)")
except ValueError as e:
    print(e)

