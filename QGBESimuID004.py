import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftfreq

# Physical constants (in arbitrary units for simulation)
hbar = 1.0
m = 1.0
G = 0.05  # Tuned for visible gravitational effects on the scale
c = 1e3   # Reduced for simulation (to make lp non-negligible; in reality ~3e8)
lp = np.sqrt(hbar * G / c**3)
beta = 41.0 / (10.0 * np.pi)
gamma = beta * lp**2  # Correction parameter

# Simulation parameters
Nx = 128  # Grid points (even for FFT)
L = 20.0  # Box size [-L/2, L/2]
dx = L / Nx
dt = 0.01  # Time step
nt_steps = 200  # Number of time steps
snapshots = [0, 50, 100, 150]  # Snapshot indices

# Grid
x = np.linspace(-L/2, L/2, Nx, endpoint=False)
X1, X2 = np.meshgrid(x, x, indexing='ij')  # X1: rows (x1), X2: columns (x2)

# Initial condition: Two separated 1D Gaussians for particles
sigma = 0.5
d = 6.0  # Initial separation
A = (1.0 / (np.pi * sigma**2)) ** 0.25  # Normalization for 1D Gaussian
psi0 = A * np.exp( -(X1 + d/2)**2 / (2 * sigma**2) ) * \
       A * np.exp( -(X2 - d/2)**2 / (2 * sigma**2) )
# Normalize
psi0 /= np.sqrt(np.sum(np.abs(psi0)**2) * dx**2)

def compute_density_and_potential(psi, correction=True):
    abs_psi_sq = np.abs(psi)**2
    rho = 2 * m * np.sum(abs_psi_sq, axis=1) * dx  # Marginal density
    kernel_factor = gamma if correction else 0.0
    Xi, Xj = np.meshgrid(x, x, indexing='ij')
    r = np.abs(Xi - Xj)
    mask = r > 1e-10
    K = np.zeros_like(r)
    K[mask] = 1.0 / r[mask] * (1.0 + kernel_factor / r[mask]**2)
    Phi = -G * np.sum(K * rho[np.newaxis, :], axis=1) * dx
    return rho, Phi

def apply_potential(psi, Phi, dt):
    Phi_x1 = np.tile(Phi[:, np.newaxis], (1, Nx))
    Phi_x2 = np.tile(Phi[np.newaxis, :], (Nx, 1))
    V = m * (Phi_x1 + Phi_x2)
    psi *= np.exp(-1j * V * dt / hbar)
    return psi

def free_evolve(psi, dt):
    k = 2 * np.pi * fftfreq(Nx, dx)
    KX, KY = np.meshgrid(k, k, indexing='ij')
    psi_hat = fft2(psi)
    phase = np.exp(-1j * hbar * (KX**2 + KY**2) * dt / (2 * m))
    psi_hat *= phase
    return ifft2(psi_hat)

def split_step_step(psi, dt, correction=True):
    psi = free_evolve(psi, dt / 2)
    _, Phi = compute_density_and_potential(psi, correction)
    psi = apply_potential(psi, Phi, dt)
    psi = free_evolve(psi, dt / 2)
    norm = np.sqrt(np.sum(np.abs(psi)**2) * dx**2)
    psi /= norm
    return psi

def simulate(with_correction=True):
    psi = psi0.copy()
    history = [psi.copy()]
    for step in range(1, nt_steps + 1):
        psi = split_step_step(psi, dt, with_correction)
        if step in snapshots[1:]:
            history.append(psi.copy())
    return history

# Run simulations
print("Running simulation without correction...")
history_no_corr = simulate(with_correction=False)
print("Final norm without correction:", np.sum(np.abs(history_no_corr[-1])**2) * dx**2)

print("Running simulation with correction...")
history_with_corr = simulate(with_correction=True)
print("Final norm with correction:", np.sum(np.abs(history_with_corr[-1])**2) * dx**2)

# Save the simulated data to NPZ file for download
data = {
    'x': x,
    'dx': dx,
    'dt': dt,
    'snapshots': np.array(snapshots),
    'history_no_corr_real': [h.real for h in history_no_corr],
    'history_no_corr_imag': [h.imag for h in history_no_corr],
    'history_with_corr_real': [h.real for h in history_with_corr],
    'history_with_corr_imag': [h.imag for h in history_with_corr]
}
np.savez('bridge_simulation_data.npz', **data)
print("Data saved to 'bridge_simulation_data.npz'")

# Generate and save the plot
fig, axs = plt.subplots(2, len(snapshots), figsize=(15, 6))
titles = ['t=0', 't=0.5', 't=1.0', 't=1.5']
for i, t in enumerate(titles):
    # Without correction
    density_no = np.abs(history_no_corr[i])**2
    im1 = axs[0, i].imshow(density_no, extent=[x[0], x[-1], x[0], x[-1]], origin='lower', cmap='hot', aspect='auto')
    axs[0, i].set_title(f'Without Correction ({t})')
    axs[0, i].set_xlabel('x1')
    axs[0, i].set_ylabel('x2')
    plt.colorbar(im1, ax=axs[0, i])

    # With correction
    density_with = np.abs(history_with_corr[i])**2
    im2 = axs[1, i].imshow(density_with, extent=[x[0], x[-1], x[0], x[-1]], origin='lower', cmap='hot', aspect='auto')
    axs[1, i].set_title(f'With Correction ({t})')
    axs[1, i].set_xlabel('x1')
    axs[1, i].set_ylabel('x2')
    plt.colorbar(im2, ax=axs[1, i])

plt.tight_layout()
plt.savefig('bridge_simulation.png', dpi=150)
plt.show()
print("Plot saved to 'bridge_simulation.png'")
