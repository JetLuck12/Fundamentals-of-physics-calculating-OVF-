# File: nls_solver.py

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.fft import fft, ifft, fftfreq

# Параметры задачи
L = 10          # Граница по пространству
N = 256         # Количество точек по пространству
dx = 2 * L / N  # Шаг по пространству
x = np.linspace(-L, L, N, endpoint=False)  # Пространственная сетка
dt_cfl = dx**2 / 2  # Максимально допустимый шаг по времени
print(f"Максимально допустимый шаг времени (CFL): {dt_cfl:.5f}")
dt_stable = dt_cfl /100  # Шаг по времени (стабильный)
t_max = 4.0       # Максимальное время

dt_unstable = 300 * dt_cfl

def split_step_nls(dt, t_max, lam, c):
    """ Решение НУШ методом расщепления """
    t_steps = int(t_max / dt)
    k = fftfreq(N, d=dx) * 2 * np.pi  # Частоты для Фурье
    k2 = k ** 2

    # Начальное условие
    A = initial_condition(x, lam, c).astype(complex)
    A_ft = fft(A)

    # Эволюция по времени
    A_t = np.zeros((t_steps, N), dtype=complex)
    for n in range(t_steps):
        A_t[n] = A
        # Нелинейный шаг
        A *= np.exp(1j * dt * np.abs(A) ** 2)
        # Линейный шаг
        A_ft = fft(A)
        A_ft *= np.exp(-1j * dt * k2)
        A = ifft(A_ft)
    return A_t

def initial_condition(x, lam, c):
    """ Начальное условие: cλ / cosh(λx) """
    return c * lam / np.cosh(lam * x)

# Функция для построения графиков
def plot_3d_solution(A_t, title, t_max, dt):
    t_steps = A_t.shape[0]
    time = np.linspace(0, t_max, t_steps)
    X, T = np.meshgrid(x, time)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, T, np.abs(A_t), cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('|A(x, t)|')
    plt.show()

# Main function
if __name__ == "__main__":
    #A_stable = split_step_nls(dt_stable, t_max, lam=2, c=1)
    A_unstable = split_step_nls(dt_unstable, t_max, lam=2, c=1)
    #A_stable_alt = split_step_nls(dt_stable, t_max, lam=3, c=0.8)

    #plot_3d_solution(A_stable, "Устойчивое решение (λ=2, c=1)", t_max, dt_stable)
    plot_3d_solution(A_unstable, "Неустойчивое решение (увеличенный шаг dt)", t_max, dt_unstable)
    #plot_3d_solution(A_stable_alt, "Дополнительное устойчивое решение (λ=3, c=0.8)", t_max, dt_stable)
