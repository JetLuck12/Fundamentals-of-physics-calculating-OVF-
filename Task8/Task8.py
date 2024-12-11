import numpy as np
import matplotlib.pyplot as plt

x = 0.001

def f(u, v):
    u_prime = 998 * u + 1998 * v
    v_prime = -999 * u - 1999 * v
    return u_prime, v_prime

def f_exact(x):
    u_exact = 2 * np.exp(-x) - np.exp(-1000*x)
    v_exact = - np.exp(-x) + np.exp(-1000*x)
    return u_exact, v_exact

def explicit_euler(u0, v0, h, N):
    u, v = u0, v0
    us, vs = np.ndarray(N), np.ndarray(N)
    for i in range(N):
        us[i] = u
        vs[i] = v
        u_prime, v_prime = f(u, v)
        u += h * u_prime
        v += h * v_prime
    return us, vs

def semi_implicit(u0, v0, h, N):
    u, v = u0, v0
    us, vs = np.ndarray(N), np.ndarray(N)
    for i in range(N):
        us[i] = u
        vs[i] = v
        u_mid, v_mid = u + h/2 * f(u, v)[0], v + h/2 * f(u, v)[1]
        u_prime, v_prime = f(u_mid, v_mid)
        u += h * u_prime
        v += h * v_prime
    return us, vs

def implicit_euler(u0, v0, h, N):
    u, v = u0, v0
    us, vs = np.ndarray(N), np.ndarray(N)
    for i in range(N):
        us[i] = u
        vs[i] = v
        u_next, v_next = u + h * f(u, v)[0], v + h * f(u, v)[1]
        u_prime, v_prime = f(u_next, v_next)
        u += h * u_prime
        v += h * v_prime
    return us, vs

u0, v0 = 1, 0
h = 0.001

explicit_errors = []
semi_implicit_errors = []
implicit_errors = []
n_s = []


for i in range(7):
    N = 10**i
    n_s.append(N)
    u_exact, v_exact = np.ndarray(N), np.ndarray(N)

    for i in range(N):
        u,v = f_exact(h*i)
        u_exact[i] = u
        v_exact[i] = v


    us_explicit, vs_explicit = explicit_euler(u0, v0, h, N)
    us_semi_implicit, vs_semi_implicit = semi_implicit(u0, v0, h, N)
    us_implicit, vs_implicit = implicit_euler(u0, v0, h, N)

    explicit_errors.append(max(max(abs(us_explicit-u_exact)), max(abs(vs_explicit-v_exact))))
    semi_implicit_errors.append(max(max(abs(us_semi_implicit-u_exact)), max(abs(vs_semi_implicit-v_exact))))
    implicit_errors.append(max(max(abs(us_implicit-u_exact)), max(abs(vs_implicit-v_exact))))

plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.plot(n_s, explicit_errors)
plt.title('Явная схема')
plt.ylabel('u и v')
plt.yscale('symlog')
plt.xscale('log')
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(n_s, semi_implicit_errors)
plt.title('Полунеявная схема')
plt.ylabel('u и v')
plt.yscale('symlog')
plt.xscale('log')
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(n_s, implicit_errors)
plt.title('Неявная схема Эйлера')
plt.ylabel('u и v')
plt.yscale('symlog')
plt.xscale('log')
plt.grid()
plt.tight_layout()
plt.show()
