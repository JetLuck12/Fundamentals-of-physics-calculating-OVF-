def bisection_method(func, U0, a, E_min, E_max, tol):
    E_low = E_min
    E_high = E_max

    while (E_high - E_low) / 2 > tol:
        E_mid = (E_low + E_high) / 2
        f_low = func(E_low, U0, a)
        f_mid = func(E_mid, U0, a)

        if f_low * f_mid <= 0:
            E_high = E_mid
        else:
            E_low = E_mid

    return (E_low + E_high) / 2