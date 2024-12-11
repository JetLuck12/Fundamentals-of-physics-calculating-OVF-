import numpy as np

def iterate(E, U0, a):
    if E >= 0 or E <= -U0:
        pass
    else:
        k = np.sqrt(2 * (E + U0))
        if np.tan(k * a) != 0:
            return -U0 / (1 + 1 / (np.tan(k * a) ** 2))
    return E

def iterate_reverse(E, U0, a):
    if E >= 0 or E <= -U0:
        pass
    else:
        return (np.atan(np.sqrt(-E/(U0+E)))**2)/(2*a**2) - U0
    return E

def iterate_multiplier(E, U0, a, l):
    if E >= 0 or E <= -U0:
        pass
    else:
        k = np.sqrt(2 * (E + U0))
        kappa = np.sqrt(2 * np.abs(E))
        if np.tan(k * a) != 0:
            f_res = (k * np.tan(k * a) - kappa)
            return E - l*f_res, f_res
    return E

def simple_iteration_method(U0, a, E_init, tol, max_iter):
    E_old = E_init
    k_old = np.sqrt(2 * (E_old + U0))
    kappa_old = np.sqrt(2 * np.abs(E_old))
    f_res_old = (k_old * np.tan(k_old * a) - kappa_old)

    E_new, f_res_new = iterate_multiplier(E_old, U0, a, -1)
    b, f_res_new = iterate_multiplier(E_new, U0, a, -1)

    df = (E_new - E_old) / (f_res_new - f_res_old)
    l = (df / abs(df)) * df
    E_old = E_new
    for i in range(max_iter):
        E_new, f_res_new = iterate_multiplier(E_old, U0, a,l)
        df = (E_new-E_old)/(f_res_new - f_res_old)
        l = (df/abs(df))*df
        if np.abs(E_new - E_old) < tol:
            return E_new
        E_old = E_new
    raise ValueError("Метод не сошелся за максимальное количество итераций")