# File: nls_solver.py

import numpy as np
import pyfftw
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants
L = 10            # Domain size [-L, L]
N = 256           # Number of spatial points
T = 5             # Total simulation time
dt = 0.01         # Time step
lambda_val = 2    # λ value
c_values = [0.5, 1, 2]  # Different c values

# Define the spatial domain
x = np.linspace(-L, L, N, endpoint=False)  # Spatial points
dx = x[1] - x[0]  # Spatial resolution
k = 2 * np.pi * np.fft.fftfreq(N, d=dx)  # Wavenumbers

# Split-step Fourier method implementation
def split_step_fourier(A_init, dt, T, k, x):
    """
    Solves the nonlinear Schrödinger equation using split-step Fourier method.
    A_init: Initial condition
    dt: Time step
    T: Total simulation time
    k: Wavenumber array
    x: Spatial array
    """
    n_steps = int(T / dt)  # Number of time steps
    A = A_init.copy()
    results = [np.abs(A)]  # Store |A(x,t)| for visualization

    for _ in range(n_steps):
        # Step 1: Nonlinear part in real space
        A = A * np.exp(1j * 2 * dt * np.abs(A)**2)

        # Step 2: Linear part in Fourier space
        A_hat = pyfftw.interfaces.numpy_fft.fft(A)  # FFT
        A_hat *= np.exp(-1j * (k**2) * dt)  # Linear phase factor
        A = pyfftw.interfaces.numpy_fft.ifft(A_hat)  # Inverse FFT

        results.append(np.abs(A))  # Store |A(x,t)|

    return np.array(results)

# Initial condition generator
def initial_condition(lambda_val, c, x):
    """
    Generates the initial condition A(x, 0).
    """
    return (c * lambda_val) / np.cosh(lambda_val * x)

# Main function
if __name__ == "__main__":
    fig = plt.figure(figsize=(12, 8))

    for i, c in enumerate(c_values):
        # Initial condition
        A0 = initial_condition(lambda_val, c, x)

        # Solve the equation
        A_results = split_step_fourier(A0, dt, T, k, x)

        # Visualization
        ax = fig.add_subplot(1, len(c_values), i + 1, projection="3d")
        X, T_grid = np.meshgrid(x, np.linspace(0, T, A_results.shape[0]))
        ax.plot_surface(X, T_grid, A_results, cmap="viridis")
        ax.set_title(f"|A(x,t)| for c = {c}")
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        ax.set_zlabel("|A(x,t)|")

    plt.tight_layout()
    plt.show()
