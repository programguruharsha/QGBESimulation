import numpy as np
import matplotlib.pyplot as plt

# Physical constants
G = 6.67430e-11  # m^3 kg^-1 s^-2
hbar = 1.0545718e-34  # J s
c = 2.99792458e8  # m/s
lp = np.sqrt(hbar * G / c**3)  # Planck length ≈ 1.616e-35 m
beta = 41 / (10 * np.pi)  # Correction coefficient

# Grid for log10(r) and log10(m)
log_r = np.linspace(-40, 25, 200)
log_m = np.linspace(-60, 5, 200)
LogR, LogM = np.meshgrid(log_r, log_m)

# Compute r and m
R = 10**LogR
M = 10**LogM

# Dimensionless interpolation parameter eta = G m^3 r / hbar^2
# This determines regimes: eta >> 1 classical, eta << 1 quantum, ~1 biphasic
eta = G * M**3 * R / hbar**2
log_eta = np.log10(eta)

# For the effective force modification, in classical regime, it's approximately 1 + 3 beta (lp / r)^2
# But here, we contour log_eta as the regime interpolator, and shade based on it
# The force modification is ~1 in quantum (dispersion dominated, softened), ~ 1 + 3 beta lp^2 / r^2 in classical

# Create figure
fig, ax = plt.subplots(figsize=(10, 8))

# Contour plot of log10(eta)
levels = np.linspace(-20, 5, 21)  # Adjust levels to cover relevant range
cf = ax.contourf(LogR, LogM, log_eta, levels=levels, cmap='RdYlBu_r', extend='both')
plt.colorbar(cf, ax=ax, label=r'$\log_{10} \eta$, $\eta = G m^3 r / \hbar^2$')

# Shade regions
# Quantum regime: eta < 0.1
quantum_mask = log_eta < np.log10(0.1)
ax.contourf(LogR, LogM, quantum_mask.astype(float), levels=[0.5, 1.5], colors='blue', alpha=0.3, hatches=['//'])

# Biphasic: 0.1 < eta < 10
biphasic_mask = (log_eta > np.log10(0.1)) & (log_eta < np.log10(10))
ax.contourf(LogR, LogM, biphasic_mask.astype(float), levels=[0.5, 1.5], colors='purple', alpha=0.3, hatches=['\\\\'])

# Classical: eta > 10
classical_mask = log_eta > np.log10(10)
ax.contourf(LogR, LogM, classical_mask.astype(float), levels=[0.5, 1.5], colors='red', alpha=0.3, hatches=['++'])

# Loop corrections dominate: where beta lp^2 / r^2 > 1, i.e., r < sqrt(beta) lp ≈ lp
r_loop = np.sqrt(beta) * lp
log_r_loop = np.log10(r_loop)  # ≈ -34.2
ax.axvline(log_r_loop, color='black', linestyle='--', alpha=0.7, label=r'$r \approx l_p$ (Loop corrections)')

# Another limit from figure: r_c = 1.7e-31 m
r_limit = 1.7e-31
log_r_limit = np.log10(r_limit)  # ≈ -30.77
ax.axvline(log_r_limit, color='gray', linestyle=':', alpha=0.7, label=r'$r_c \approx 1.7 \times 10^{-31}$ m')

# Boundary contour for eta=1 (transition)
ax.contour(LogR, LogM, log_eta, levels=[0], colors='black', linestyles='-', linewidth=1.5, alpha=0.8)

# Labels for regions
ax.text(0, 0, 'Classical', ha='center', va='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
ax.text(-20, -40, 'Quantum\nDispersion', ha='center', va='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='blue', alpha=0.3))
ax.text(-10, -20, 'Biphasic', ha='center', va='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='purple', alpha=0.3))

# Example points from figure
# H Atom: approx m ~ 1 ug = 1e-9 kg, log_m=-9; r~1.7e-31 m, log_r≈-30.8
ax.plot(-30.8, -9, 'ko', markersize=8, label='H Atom')

# FDM Halo: m < 1.8e-35 kg, say point at m=1e-35, r= some, but approx log_m=-35, log_r=0 (arbitrary for halo scale)
ax.plot(0, -35, 'gs', markersize=8, label='FDM Halo')

# Plot limits
ax.set_xlim(-40, 25)
ax.set_ylim(-60, 5)
ax.set_xlabel(r'$\log_{10} r$ (Length Scale)')
ax.set_ylabel(r'$\log_{10} m$ (Mass, kg)')
ax.set_title('Regime Interpolation: Classical to Quantum Transition in BRIDGE Framework')
ax.legend()
ax.grid(True, alpha=0.3)

# Simulate varying hbar artificially: to show interpolation, we can plot for different effective hbar
# For example, hbar_eff = hbar * factor, but for now, the plot shows hbar->0 as eta -> infinity (classical)

plt.tight_layout()
plt.show()

# To artificially vary hbar for interpolation demonstration:
# For hbar_eff = hbar * 10**k, log_eta shifts by -2k
# E.g., for k=-3 (smaller hbar, more classical), levels shift up by 6 in log_eta
print("Simulation complete. Run the code to visualize the phase diagram.")
print("The contours illustrate smooth transitions: as hbar -> 0, the classical region expands.")
