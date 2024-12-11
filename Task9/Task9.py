import numpy as np
import matplotlib.pyplot as plt

def exact_func(x):
    return -np.cos(x)

def solve_tridiagonal_system(a, b, c, d):
    n = len(d)
    c_prime = np.zeros(n - 1)
    d_prime = np.zeros(n)

    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]

    for i in range(1, n - 1):
        denom = b[i] - a[i - 1] * c_prime[i - 1]
        c_prime[i] = c[i] / denom
        d_prime[i] = (d[i] - a[i - 1] * d_prime[i - 1]) / denom

    d_prime[-1] = (d[-1] - a[-1] * d_prime[-2]) / (b[-1] - a[-1] * c_prime[-2])

    x = np.zeros(n)
    x[-1] = d_prime[-1]

    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]

    return x


def solve_bvp(boundary_condition_type, N):
    x_start = -np.pi / 2
    x_end = np.pi / 2
    h = (x_end - x_start) / (N - 1)

    x = np.linspace(x_start, x_end, N)

    a = np.ones(N - 1)
    b = -2 * np.ones(N)
    c = np.ones(N - 1)
    d = h ** 2 * np.cos(x)

    if boundary_condition_type == "dirichlet":
        d[0] = 0
        d[-1] = 0

    elif boundary_condition_type == "neumann":
        d[0] += 1 * h ** 2
        d[-1] += 1 * h ** 2

    elif boundary_condition_type == "mixed":
        d[0] = 0
        d[-1] += 1 * h ** 2

    y = solve_tridiagonal_system(a, b, c, d)

    if boundary_condition_type == "neumann":
        y[0] = y[1]
        y[-1] = y[-2]
    elif boundary_condition_type == "mixed":
        y[-1] = y[-2]

    return x, y


def plot_solution(x, y, boundary_condition_type,K):

    plt.subplot(3,1,K)
    plt.plot(x, y, label=boundary_condition_type)
    plt.title(f"Решение задачи при {boundary_condition_type} граничных условиях")
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.grid(True)
    plt.legend()


def main():
    boundary_conditions = ["Дирихле", "Нейман", "Смешанные"]
    Z = 1
    plt.figure(figsize = (12,8))
    x_start = -np.pi / 2
    x_end = np.pi / 2

    Z = 1
    for bc_type in boundary_conditions:
        errors = []
        Splits = []

        for i in range(1,6):
            N = 10 ** i
            Splits.append(N)
            x_exact = np.linspace(x_start, x_end, N)
            y_exact = exact_func(x_exact)
            x, y = solve_bvp(bc_type, N)
            errors.append(max(abs(y-y_exact)))
        plt.subplot(3,1,Z)
        plt.yscale('log')
        plt.xscale('log')
        plt.plot(Splits, errors, label = f"{bc_type}")
        Z += 1
        plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
