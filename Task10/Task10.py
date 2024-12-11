import numpy as np
import matplotlib.pyplot as plt


L = 1.0
T = 0.1
alpha = 1.0


def crank_nicolson(N, M):
    h = L / (N - 1)
    dt = T / M
    alpha = dt / h ** 2


    x = np.linspace(0, L, N)


    u = np.sin(np.pi * x)


    A = np.zeros((N - 2, N - 2))
    B = np.zeros((N - 2, N - 2))


    for i in range(N - 2):
        if i > 0:
            A[i, i - 1] = -alpha / 2
            B[i, i - 1] = alpha / 2
        A[i, i] = 1 + alpha
        B[i, i] = 1 - alpha
        if i < N - 3:
            A[i, i + 1] = -alpha / 2
            B[i, i + 1] = alpha / 2


    def thomas_algorithm(A, d):
        n = len(d)
        c_prime = np.zeros(n - 1)
        d_prime = np.zeros(n)

        c_prime[0] = A[0, 1] / A[0, 0]
        d_prime[0] = d[0] / A[0, 0]

        for i in range(1, n - 1):
            denom = A[i, i] - A[i, i - 1] * c_prime[i - 1]
            c_prime[i] = A[i, i + 1] / denom
            d_prime[i] = (d[i] - A[i, i - 1] * d_prime[i - 1]) / denom

        d_prime[-1] = (d[-1] - A[-1, -2] * d_prime[-2]) / (A[-1, -1] - A[-1, -2] * c_prime[-2])

        x = np.zeros(n)
        x[-1] = d_prime[-1]

        for i in range(n - 2, -1, -1):
            x[i] = d_prime[i] - c_prime[i] * x[i + 1]

        return x

    for n in range(1, int(M)):

        d = B @ u[1:-1]


        u[1:-1] = thomas_algorithm(A, d)

    return u, x


def convergence_test():
    N_values = [2**i for i in range(3,9)]

    errors = []
    second_order_func = []
    h_values = []

    for N in N_values:

        M = T / (alpha*(L/(N-1))**2)
        u_num, x = crank_nicolson(N, M)

        # Аналитическое решение
        u_exact = np.exp(-np.pi ** 2 * T) * np.sin(np.pi * x)

        # Вычисление ошибки
        error = max(abs(u_num - u_exact))
        errors.append(error)
        second_order_func.append((L / (N - 1))**2)

        # Шаг сетки
        h_values.append(L / (N - 1))

    # Построение зависимости ошибки от шага по пространству
    plt.figure(figsize = (12,6))
    plt.plot(h_values, errors, marker='o', label="Ошибка")
    plt.plot(h_values, second_order_func, marker='o', label="Ошибка второго порядка")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Шаг сетки h")
    plt.ylabel("Ошибка")
    plt.title("Сходимость схемы Кранка-Николсон")
    plt.grid(True)
    plt.legend()
    plt.show()


# Запуск теста на сходимость
convergence_test()
