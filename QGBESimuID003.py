import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Constants
G = 6.6743e-11  # m^3 kg^-1 s^-2
hbar = 1.0546e-34  # J s
c = 2.99792458e8  # m/s
lp = np.sqrt(hbar * G / c**3)  # Planck length

# Grid for log scales
log_r_min, log_r_max = -40, 25
log_m_min, log_m_max = -60, 5
n_points = 100
log_r = np.linspace(log_r_min, log_r_max, n_points)
log_m = np.linspace(log_m_min, log_m_max, n_points)
LogR, LogM = np.meshgrid(log_r, log_m)
R = 10**LogR
M = 10**LogM
gamma = G * M**3 * R / hbar**2
log_gamma = np.log10(gamma)

# Create figure
fig, ax = plt.subplots(figsize=(16, 9), facecolor='white')
plt.rcParams['font.family'] = 'DejaVu Sans'

# Contour plot
levels = np.linspace(-4, 4, 9)
cf = ax.contourf(LogR, LogM, log_gamma, levels=levels, cmap='coolwarm', extend='both')
contour_lines = ax.contour(LogR, LogM, log_gamma, levels=[-2, -1, 0, 1, 2], colors='black', linewidths=1.5)
ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%1.0f')

# Colorbar
cbar = plt.colorbar(cf, ax=ax, shrink=0.8, aspect=20)
cbar.set_label(r'$\log_{10}(\gamma)$', fontsize=10)
cbar.set_ticks([-3, -2, -1, 0, 1, 2, 3])
cbar.ax.text(1.1, -0.05, 'Quantum Dispersion ($\\gamma \\ll 1$)', transform=cbar.ax.transAxes, fontsize=9, style='italic', ha='left')
cbar.ax.text(1.1, 1.05, 'Classical Collapse ($\\gamma \\gg 1$)', transform=cbar.ax.transAxes, fontsize=9, style='italic', ha='left')

# Labels and title
ax.set_xlabel('Length Scale $r$ (m)', fontsize=10)
ax.set_ylabel('Mass $m$ (kg)', fontsize=10)
ax.set_title('Regime Interpolation: Classical to Quantum Transition in the BRIDGE Framework', fontsize=14, fontweight='bold', pad=20)

# Ticks
ax.set_xticks(np.arange(-40, 26, 10))
ax.set_yticks(np.arange(-60, 6, 10))
ax.grid(True, alpha=0.3, linestyle='--')

# Transition line highlight
trans_line = ax.contour(LogR, LogM, log_gamma, levels=[0], colors='black', linestyles='dashed', linewidths=2.5)
ax.text(-10, -10, r'Classical-Quantum Transition ($\hbar \to 0$ Limit)', fontsize=9, style='italic', ha='center', rotation=-30)

# Biphasic region shading: log_gamma <0
cf_hatch = ax.contourf(LogR, LogM, log_gamma, levels=[-np.inf, 0], hatches=['////'], colors='lightblue', alpha=0.2)

# Label for biphasic
ax.text( -10, -30, 'Biphasic Quantum Regime (e.g., Fuzzy Dark Matter Halos)', fontsize=10, style='italic', 
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.5), ha='center', va='center', rotation=30)

# Key points
# 1. FDM Halo
m_fdm = 1.8e-58
r_fdm = 3e22
log_m_fdm = np.log10(m_fdm)
log_r_fdm = np.log10(r_fdm)
ax.plot(log_r_fdm, log_m_fdm, 'ko', markersize=6)
ax.annotate(r'FDM Halo: $m \approx 1.8\times10^{-58}$ kg, $r \approx 3\times10^{22}$ m, $\log_{10}(\gamma) \approx -93$', 
            xy=(log_r_fdm, log_m_fdm), xytext=(10, -20), fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white'),
            arrowprops=dict(arrowstyle='->', color='black'))

# 2. H Atom
m_h = 1.7e-27
r_h = 1e-10
log_m_h = np.log10(m_h)
log_r_h = np.log10(r_h)
ax.plot(log_r_h, log_m_h, 'ko', markersize=6)
ax.annotate(r'H Atom: $m \approx 1.7\times10^{-27}$ kg, $r \approx 10^{-10}$ m, $\log_{10}(\gamma) \approx -33$', 
            xy=(log_r_h, log_m_h), xytext=(-20, -40), fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white'),
            arrowprops=dict(arrowstyle='->', color='black'))

# 3. 1 ug particle
m_ug = 1e-9
r_c_ug = hbar**2 / (G * m_ug**3)
log_r_c_ug = np.log10(r_c_ug)
log_m_ug = np.log10(m_ug)
ax.plot(log_r_c_ug, log_m_ug, 'ko', markersize=6)
ax.annotate(r'1 $\mu$g: $m = 10^{-9}$ kg, $r_c \approx 1.7\times10^{-31}$ m, $\log_{10}(\gamma) = 0$', 
            xy=(log_r_c_ug, log_m_ug), xytext=(-30, 0), fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white'),
            arrowprops=dict(arrowstyle='->', color='black'))

# Planck scale line
log_lp = np.log10(lp)
ax.axvline(log_lp, color='gray', linestyle=':', alpha=0.7)
ax.annotate(r'$l_p \approx 1.6\times10^{-35}$ m (Loop Corrections Dominate)', 
            xy=(log_lp, -50), xytext=(log_lp+5, -50), fontsize=9,
            arrowprops=dict(arrowstyle='->', color='gray'))

# Inset boxes
# Classical inset upper left
ax.add_patch(FancyBboxPatch(xy=(-35, -10), width=15, height=12, boxstyle="round,pad=0.1", 
                              facecolor='white', edgecolor='black', linewidth=2))
ax.text(-27.5, -4, r'$\gamma = \frac{G m^3 r}{\hbar^2} \gg 1$, $V_g \approx -\frac{G m^2}{r}$ (point-like)', 
        fontsize=9, ha='center', va='center')

# Quantum inset lower right
ax.add_patch(FancyBboxPatch(xy=(10, -55), width=12, height=10, boxstyle="round,pad=0.1", 
                              facecolor='white', edgecolor='black', linewidth=2))
ax.text(16, -50, r'$\gamma \ll 1$, $|\psi|^2$ spreads, $F_{eff}$ softened by delocalization + $\beta l_p^2 / r^3$ correction negligible except near $l_p$', 
        fontsize=8, ha='left', va='center')

# Error bands: approximate by filling between +/-0.5 dex contours
log_gamma_upper = log_gamma + 0.5
log_gamma_lower = log_gamma - 0.5
ax.contourf(LogR, LogM, log_gamma_upper, levels=[-0.5, 0.5], colors='gray', alpha=0.1)
ax.contourf(LogR, LogM, log_gamma_lower, levels=[-0.5, 0.5], colors='gray', alpha=0.1)

# Caption
plt.figtext(0.5, 0.01, r"Fig. 6: Phase diagram in log-log ($r$, $m$) plane showing regime interpolation via nonlinearity parameter $\gamma$, with contours illustrating smooth $\hbar \to 0$ classical recovery (red, $\gamma \gg 1$) and quantum-dominated biphasic regions (blue, $\gamma \ll 1$) for wave function collapse. Shading highlights $\hbar$ effects in fuzzy dark matter halos and optomechanical tests; simulations vary $\hbar$ artificially via $\gamma$ scaling for interpolation ($G = 6.67\times10^{-11}$, $\hbar = 1.05\times10^{-34}$, $\beta \approx 1.30$).", 
            ha='center', fontsize=9, style='italic')

# Tight layout
plt.tight_layout()

# To display or save
# plt.show()
plt.savefig('phase_diagram_bridge.png', dpi=300, bbox_inches='tight')
