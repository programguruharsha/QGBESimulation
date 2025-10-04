import numpy as np
import matplotlib.pyplot as plt

# Constants (normalized: we plot V(r) / (G m) for unitless plot, as scales are logarithmic)
beta = 41 / (10 * np.pi)  # ≈1.303
lp = 1.616e-35  # Planck length in meters
delta_beta = 0.1  # Variant for error bands (e.g., uncertainty in beta)

# Range of r: from well below lp to cosmological scales
r_min = 1e-40
r_max = 1e10
r = np.logspace(np.log10(r_min), np.log10(r_max), 1000)

# Classical potential: V_class(r) / (G m) = -1 / r
V_class = -1 / r

# Corrected potential: V_corr(r) / (G m) = -1/r * (1 + beta * lp**2 / r**2)
V_corr = -1 / r * (1 + beta * lp**2 / r**2)

# Variants for error bands
beta_upper = beta + delta_beta
beta_lower = beta - delta_beta
V_corr_upper = -1 / r * (1 + beta_upper * lp**2 / r**2)
V_corr_lower = -1 / r * (1 + beta_lower * lp**2 / r**2)

# Create log-log plot
plt.figure(figsize=(10, 6))
plt.loglog(r, np.abs(V_class), 'k--', linewidth=2, label='Classical: $-G m / r$')
plt.loglog(r, np.abs(V_corr), 'b-', linewidth=2, label=f'Corrected: $-G m / r (1 + \\beta l_p^2 / r^2)$ ($\\beta \\approx {beta:.3f}$)')
plt.fill_between(r, np.abs(V_corr_lower), np.abs(V_corr_upper), alpha=0.3, color='blue', label=f'Error band ($\\beta \\pm {delta_beta}$)')
plt.axvline(x=lp, color='red', linestyle=':', linewidth=1, label=f'Planck scale $l_p \\sim 10^{{-35}}$ m')

# Labels and title
plt.xlabel('$r$ (m)', fontsize=12)
plt.ylabel('$|V(r)| / (G m)$ (unitless)', fontsize=12)
plt.title('Log-Log Plot of Gravitational Potential: Classical vs. Quantum-Corrected BRIDGE Model', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, which='both', alpha=0.3)
plt.xlim(r_min, r_max)
plt.ylim(1e-20, 1e5)  # Adjust y-limits to focus on relevant range

# Annotations
plt.annotate('Quantum effects dominate\n(small $r$, divergence $\\sim 1/r^3$)', 
             xy=(1e-30, 1e3), xytext=(1e-35, 1e4),
             arrowprops=dict(arrowstyle='->', color='blue'), fontsize=10)
plt.annotate('Classical recovery\n(large $r$)', 
             xy=(1e5, 1e-5), xytext=(1e8, 1e-6),
             arrowprops=dict(arrowstyle='->', color='black'), fontsize=10)

# Save the figure (for inclusion in document)
plt.savefig('bridge_potential_plot.png', dpi=300, bbox_inches='tight')
plt.show()
