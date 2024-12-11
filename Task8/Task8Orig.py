from cProfile import label

import numpy as np
import matplotlib.pyplot as plt



J = np.empty((2, 2))
J[0][0] = 999
J[0][1] = 1998
J[1][0] = -999
J[1][1] = -1999

u0, v0 = 1, 0
x = 1

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

def HalfImplicitScheme(A, xArr, y0):
    yArr = np.empty((xArr.size, y0.size))
    yArr[0] = y0

    E = np.eye(y0.size)

    for i in range(1, xArr.size):
        yPrev = yArr[i - 1]

        h = xArr[i] - xArr[i - 1]
        yArr[i] = yPrev + np.linalg.inv(E - h * A) @ (h * A @ yPrev) / 2 + (h * A @ yPrev) / 2

    return yArr


def ImplicitScheme(A, xArr, y0):
    yArr = np.empty((xArr.size, y0.size))
    yArr[0] = y0

    E = np.eye(y0.size)

    for i in range(1, xArr.size):
        yPrev = yArr[i - 1]

        h = xArr[i] - xArr[i - 1]
        yArr[i] = yPrev + np.linalg.inv(E - h * A) @ (h * A @ yPrev)

    return yArr

def plot_explicit():


    us_explicit, vs_explicit = explicit_euler(u0, v0, h, N)

    plt.figure(figsize=(12, 6))

    t = np.linspace(0, N*h, N)

    plt.subplot(3, 1, 1)
    plt.plot(t, abs(us_explicit-u_exact))
    plt.plot(t, abs(vs_explicit-v_exact))
    plt.title('Явная схема')
    plt.ylabel('u и v')
    plt.yscale('log')
    plt.grid()

def SolveHalfImplicit(h, xMax, A):
    xArr = np.arange(0, xMax, h)
    y0 = np.array([u0, v0])

    yCalculated = HalfImplicitScheme(A, xArr, y0)
    uCalculated = yCalculated[:, 0]
    vCalculated = yCalculated[:, 1]
    return uCalculated, vCalculated

def SolveAndPlotHalfImplicit(h, xMax, A):
    uCalculated, vCalculated = SolveHalfImplicit(h, xMax, A)

    plt.subplot(3,1,2)
    plt.title('Полунеявная схема')
    plt.plot(xArr, abs(uCalculated-u_exact))
    plt.plot(xArr, abs(vCalculated-v_exact))
    plt.grid()
    plt.yscale('log')

def SolveImplicit(h, xMax, A):
    xArr = np.arange(0, xMax, h)
    y0 = np.array([u0, v0])

    yCalculated = ImplicitScheme(A, xArr, y0)
    uCalculated = yCalculated[:, 0]
    vCalculated = yCalculated[:, 1]
    return uCalculated, vCalculated

def SolveAndPlotImplicit(h, xMax, A):
    uCalculated, vCalculated = SolveImplicit(h, xMax, A)

    plt.subplot(3,1,3)
    plt.title('Неявная схема')
    plt.plot(xArr, abs(uCalculated-u_exact))
    plt.plot(xArr, abs(vCalculated-v_exact))
    plt.grid()
    plt.yscale('log')

Euler_errors = []
Half_errors = []
Full_errors = []
Splits = []

for i in range(100, 600, 50):
    N = i
    Splits.append(N)
    h = 1/N
    u_exact, v_exact = np.ndarray(N), np.ndarray(N)
    for i in range(N):
        u, v = f_exact(h * i)
        u_exact[i] = u
        v_exact[i] = v
    uEuler, vEuler = explicit_euler(u0, v0, h, N)
    Euler_errors.append(max(abs(u_exact-uEuler)))
    uHalf, vHalf = SolveHalfImplicit(h, x, J)
    Half_errors.append(max(abs(u_exact-uHalf)))
    uFull, vFull = SolveImplicit(h, x, J)
    Full_errors.append(max(abs(u_exact-uFull)))

plt.plot(Splits, Euler_errors, label = "Euler")
plt.plot(Splits, Half_errors, label = "HalfImplicit")
plt.plot(Splits, Full_errors, label = "Implicit")
plt.legend()
plt.yscale('log')
#plt.xscale('log')
plt.show()

