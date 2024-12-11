import numpy as np


def newton_method(f, df, U0, a, E_init, tol, max_iter):
    e_old = E_init
    for i in range(max_iter):
        f_value = f(e_old, U0, a)
        df_value = df(e_old, U0, a)

        if np.abs(df_value) < tol:
            raise ValueError("Производная слишком мала, метод может не сходиться")

        e_new = e_old - f_value / df_value

        if np.abs(e_new - e_old) < tol:
            return e_new, i + 1

        e_old = e_new

    raise ValueError("Метод не сошелся за максимальное количество итераций")