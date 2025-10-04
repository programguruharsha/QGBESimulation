import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq

# Parameters (in units where hbar = 1)
hbar = 1.0
m = 1.0
G = 0.1  # Adjusted for reasonable collapse time
sigma = 1.0
beta = 41.0 / (10.0 * np.pi)  # ~1.303
lp = 0.1  # Planck length scaled up for visibility in simulation
tau = hbar / (G * m**(5.0/3.0) * sigma**(1.0/3.0))  # Collapse time from Page 3
print(f"Collapse time tau: {tau:.2f}")

# Grid setup (1D)
N = 512
dx = 0.2
L = N * dx
x = np.linspace(-L/2, L/2, N, endpoint=False)
k = 2 * np.pi * fftfreq(N, dx)

# Initial Gaussian wave function (normalized in 1D, ensure complex dtype)
psi0 = (1.0 / (np.pi * sigma**2))**0.25 * np.exp(-x**2 / (2.0 * sigma**2)) + 0j
# Verify normalization
norm = np.sum(np.abs(psi0)**2) * dx
print(f"Initial normalization: {norm:.4f}")

# Function to compute gravitational potential V_g(x)
def compute_Vg(psi, x, G, m, beta, lp, dx):
    N = len(x)
    rho = np.abs(psi)**2
    V = np.zeros(N)  # Default to float64, as potential is real
    for i in range(N):
        r = np.abs(x[i] - x)
        # Avoid singularity at r=0
        mask = r > 1e-8
        kernel = np.zeros(N)
        kernel[mask] = (1.0 / r[mask]) + beta * (lp**2 / r[mask]**3)
        V[i] = -G * m**2 * np.sum(rho * kernel) * dx
    return V

# Split-operator step
def split_operator_step(psi, dt, x, k, m, G, beta, lp, dx):
    # First half: potential
    V = compute_Vg(psi, x, G, m, beta, lp, dx)
    psi *= np.exp(-1j * V * dt / (2.0 * hbar))
    
    # Full kinetic in Fourier space
    psi_hat = fft(psi)
    psi_hat *= np.exp(-1j * (k**2 / (2.0 * m)) * dt * hbar / hbar)  # hbar=1
    psi = ifft(psi_hat)
    
    # Second half: potential (updated)
    V = compute_Vg(psi, x, G, m, beta, lp, dx)
    psi *= np.exp(-1j * V * dt / (2.0 * hbar))
    
    return psi

# Time evolution function
def evolve_psi(psi_init, t_final, nt, x, k, m, G, beta, lp, dx):
    dt = t_final / nt
    psi = psi_init.copy()
    psi_list = [psi.copy()]
    for _ in range(nt):
        psi = split_operator_step(psi, dt, x, k, m, G, beta, lp, dx)
        # Optional: renormalize numerically (though unitary in theory)
        norm = np.sqrt(np.sum(np.abs(psi)**2) * dx)
        psi /= norm
    return psi_list[-1]  # Return final, but we need intermediates

# To get at specific times, evolve step by step and save
def evolve_with_snapshots(psi_init, times, x, k, m, G, beta, lp, dx):
    psi = psi_init.copy()
    snapshots = {0.0: psi.copy()}
    current_t = 0.0
    dt = min(np.diff(times)) / 10.0  # Small dt for accuracy
    for target_t in times[1:]:
        while current_t < target_t:
            psi = split_operator_step(psi, dt, x, k, m, G, beta, lp, dx)
            current_t += dt
        snapshots[target_t] = psi.copy()
        # Renormalize at snapshot
        norm = np.sqrt(np.sum(np.abs(psi)**2) * dx)
        snapshots[target_t] /= norm
    return snapshots

# Times for snapshots
t_snap = [0.0, tau / 2.0, tau]

# Evolve without quantum correction (beta=0)
print("Evolving without quantum correction...")
snapshots_no_corr = evolve_with_snapshots(psi0, t_snap, x, k, m, G, 0.0, lp, dx)

# Evolve with quantum correction
print("Evolving with quantum correction...")
snapshots_with_corr = evolve_with_snapshots(psi0, t_snap, x, k, m, G, beta, lp, dx)

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Without correction
ax = axs[0]
for t, psi in snapshots_no_corr.items():
    density = np.abs(psi)**2
    ax.plot(x, density, label=f't = {t:.2f}')
ax.set_xlabel('Position x (arbitrary units)')
ax.set_ylabel(r'$|\psi(x,t)|^2$')
ax.set_title('Without Quantum Correction')
ax.legend()
ax.grid(True)

# With correction
ax = axs[1]
for t, psi in snapshots_with_corr.items():
    density = np.abs(psi)**2
    ax.plot(x, density, label=f't = {t:.2f}')
ax.set_xlabel('Position x (arbitrary units)')
ax.set_ylabel(r'$|\psi(x,t)|^2$')
ax.set_title('With Quantum Correction')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.suptitle('Wave Function Evolution Under Self-Gravity\n(Spreading then collapse due to nonlinear self-interaction)')
plt.show()

# Print final norms for verification
print(f"Final norm (no corr): {np.sum(np.abs(snapshots_no_corr[tau])**2 * dx):.4f}")
print(f"Final norm (with corr): {np.sum(np.abs(snapshots_with_corr[tau])**2 * dx):.4f}")
