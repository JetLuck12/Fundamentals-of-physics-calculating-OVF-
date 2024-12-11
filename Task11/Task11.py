import numpy as np
import matplotlib.pyplot as plt

x_min, x_max = -10, 10

def find_E(N):
    x = np.linspace(x_min, x_max, N)
    h = x[1] - x[0]
    U = 0.5 * x ** 2

    H = np.zeros((N, N))
    for i in range(N):
        H[i, i] = 1 / h**2 + U[i]
        if i > 0:
            H[i, i - 1] = -0.5 / h**2
        if i < N - 1:
            H[i, i + 1] = -0.5 / h**2

    psi = np.ones(N)
    psi /= np.linalg.norm(psi)
    mu = 0
    num_iterations = 10


    for _ in range(1,num_iterations):
        A = H - mu * np.eye(N)
        psi_new = np.linalg.solve(A, psi)
        # Нормировка
        psi_new /= np.linalg.norm(psi_new)
        psi = psi_new

    E = np.dot(psi, np.dot(H, psi)) / np.dot(psi, psi)
    return E



Ns = []
Es = []

for i in range(100,2000, 100):
    print(i)
    Ns.append(i)
    Es.append(abs(find_E(i) - 0.5))

plt.figure(figsize=(10, 5))
plt.plot(Ns, Es)
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid()
plt.show()