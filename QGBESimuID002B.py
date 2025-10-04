import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

# Set global font and style for professional look (use available sans-serif)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # Use available font to avoid warnings
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['mathtext.fontset'] = 'stix'  # For LaTeX-like math
plt.rcParams['mathtext.default'] = 'regular'  # Avoid any escape issues

# Constants
beta = 41 / (10 * np.pi)  # ≈1.303
lp = 1.616e-35  # Planck length in m
beta_lower = 1.2
beta_upper = 1.4

# Generate high-resolution r (1000 points per decade)
num_decades = np.log10(1e10) - np.log10(1e-40)
r = np.logspace(-40, 10, int(1000 * (num_decades + 1)))

# Normalized V: V(r) / (G m) = -1 / r for classical (at r=1m, V=-1)
V_class = -1 / r

# Corrected: V_corr(r) / (G m) = -1/r * (1 + beta * (lp / r)^2)
V_corr = -1 / r * (1 + beta * (lp / r)**2)

# Error bands: using beta_lower and beta_upper
V_corr_lower = -1 / r * (1 + beta_lower * (lp / r)**2)
V_corr_upper = -1 / r * (1 + beta_upper * (lp / r)**2)

# For loglog, plot absolute values (magnitudes, since V negative)
abs_V_class = np.abs(V_class)
abs_V_corr = np.abs(V_corr)
abs_V_corr_lower = np.abs(V_corr_lower)
abs_V_corr_upper = np.abs(V_corr_upper)

# Create figure: portrait, adjusted size
fig, ax = plt.subplots(figsize=(6.7, 8.3))  # Approximate 8.5x11 scaled down for single panel

# Log-log plot
ax.loglog(r, abs_V_class, color=(0.5, 0.5, 0.5), linestyle='--', linewidth=2, label='Classical: $V(r) = -Gm/r$ (dashed)')
ax.loglog(r, abs_V_corr, color=(0, 0, 1), linewidth=2.5, label='BRIDGE Corrected: $V(r) = -Gm/r [1 + \\beta l_p^2 / r^2]$ (solid)')

# Error bands: shaded light blue, opacity 0.2
ax.fill_between(r, abs_V_corr_lower, abs_V_corr_upper, color=(0.7, 0.9, 1), alpha=0.2, label='Error band ($\\beta \\in [1.2,1.4]$)')

# Planck scale vertical line: thin red dashed
ax.axvline(x=lp, color='red', linestyle=':', linewidth=1)
# Label with arrow
ax.annotate('Planck Scale $l_p$', xy=(lp, 1e4), xytext=(lp*10, 1e3),
            arrowprops=dict(arrowstyle='->', color='red', lw=1), fontsize=9, fontweight='bold',
            ha='left', va='top')

# Grid: faint light gray, major and minor
ax.grid(True, which='both', alpha=0.3, color='gray', linestyle='-')
ax.grid(True, which='minor', alpha=0.1, color='gray', linestyle=':')

# Axes labels with LaTeX (use raw strings to avoid escapes)
ax.set_xlabel(r'Distance $r$ (m)', fontsize=10)
ax.set_ylabel(r'$|V(r)|$ (arbitrary units, normalized to $Gm$ at $r=1$ m)', fontsize=10)

# Limits
ax.set_xlim(1e-40, 1e10)
ax.set_ylim(1e-5, 1e5)

# Custom ticks: specific labels (use raw for LaTeX)
ax.set_xticks([1e-40, 1e-35, 1e-10, 1e0, 1e10])
ax.set_xticklabels([r'$10^{-40}$', r'$10^{-35}$', r'$10^{-10}$', r'$10^{0}$', r'$10^{10}$'])
ax.set_yticks([1e-4, 1e-2, 1e0, 1e2, 1e4])
ax.set_yticklabels([r'$-10^{-4}$', r'$-10^{-2}$', r'$-10^{0}$', r'$-10^{2}$', r'$-10^{4}$'])

# Title: centered, bold, 14pt
plt.title('Comparison of Classical and Quantum-Corrected Gravitational Potentials', fontsize=14, fontweight='bold', pad=20)

# Legend: upper right
legend = ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=False)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_alpha(0.9)

# Inset box for beta and lp: near legend, small boxed inset (raw string)
inset_text = r'$\beta = 41/(10\pi) \approx 1.303$, $l_p = \sqrt{\hbar G / c^3} \approx 1.616 \times 10^{-35}$ m'
inset_box = FancyBboxPatch((0.68, 0.75), 0.2, 0.08, boxstyle='round,pad=0.03', 
                            facecolor='white', alpha=0.9, edgecolor='black', linewidth=0.5)
ax.add_patch(inset_box)
ax.text(0.78, 0.79, inset_text, transform=ax.transAxes, fontsize=9, style='italic',
        ha='center', va='center')

# Annotations: horizontal arrows (raw strings)
# Left: Quantum regime
ax.annotate('', xy=(1e-35, 1e3), xytext=(1e-30, 1e3),
            arrowprops=dict(arrowstyle='<|-', color='black', lw=1))
ax.text(1e-32, 1e3 * 1.5, r'Quantum Regime: Enhanced Effects (Divergence $\sim 1/r^3$)', 
        fontsize=9, style='italic', ha='center', va='bottom')

# Right: Classical recovery
ax.annotate('', xy=(1e5, 1e-3), xytext=(1e3, 1e-3),
            arrowprops=dict(arrowstyle='-|> ', color='black', lw=1))
ax.text(1e4, 1e-3 / 1.5, r'Classical Recovery: $\hbar \to 0$ Limit', 
        fontsize=9, style='italic', ha='center', va='top')

# Caption: at bottom, 9pt italic (raw string, split if needed for long text)
caption = (r'Fig. 5: Log-log plot of the gravitational potential $V(r)$ for unit masses $G m = 1$ '
           r'(arbitrary units), showing classical Newtonian form (dashed gray) and BRIDGE '
           r'quantum-corrected form (solid blue) with one-loop graviton term. Quantum effects '
           r'cause divergence at small $r \ll l_p$ (enhanced attraction), recovering classical '
           r'behavior at large $r \gg l_p$; error bands (shaded $\pm10\%$ for $\beta$ variants '
           r'from literature, e.g., 1.2-1.4) indicate theoretical uncertainty.')
fig.text(0.5, 0.01, caption, ha='center', va='bottom', fontsize=9, style='italic', wrap=True)

# Tight layout and save as SVG for vector scalability
plt.tight_layout()
plt.savefig('bridge_quantum_gravity_potential.svg', format='svg', bbox_inches='tight')
plt.savefig('bridge_quantum_gravity_potential.png', dpi=300, bbox_inches='tight')  # PNG fallback
plt.show()
