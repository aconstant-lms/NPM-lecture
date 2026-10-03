"""
Generates figures/ch4/yield_loci.pdf (Chapter 4, Section 4.3.2).
Von Mises cylinder in the space of principal stresses, sections of the von
Mises and Tresca criteria by the deviatoric (pi) plane, and the two criteria
in the tension-torsion plane (sigma, tau). Stresses in units of sigma_Y.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

blue, green, orange = "#1F5AC8", "#14963C", "#D9822B"
fig = plt.figure(figsize=(11, 3.7))

# --- (a) von Mises cylinder in principal stress space -----------------------
ax = fig.add_subplot(1, 3, 1, projection="3d")
h = np.array([1, 1, 1]) / np.sqrt(3)                   # hydrostatic axis
e1 = np.array([2, -1, -1]) / np.sqrt(6)                # basis of the pi-plane
e2 = np.array([0, 1, -1]) / np.sqrt(2)
R = np.sqrt(2 / 3)                                      # radius, sigma_Y = 1
th = np.linspace(0, 2 * np.pi, 80)
z = np.linspace(-0.6, 2.6, 2)
TH, Z = np.meshgrid(th, z)
X = [R * (np.cos(TH) * e1[i] + np.sin(TH) * e2[i]) + Z * h[i] for i in range(3)]
ax.plot_surface(*X, color=blue, alpha=0.18, linewidth=0)
for zz in (-0.6, 0.0, 2.6):
    c = [R * (np.cos(th) * e1[i] + np.sin(th) * e2[i]) + zz * h[i] for i in range(3)]
    ax.plot(*c, color=blue, lw=1.2 if zz == 0 else 0.6)
ax.plot(*[np.array([-0.8, 3.0]) * h[i] for i in range(3)], color="k", lw=1)
ax.text(*(3.05 * h), "hydrostatic axis", fontsize=7)
for i, lab in enumerate([r"$\sigma_1$", r"$\sigma_2$", r"$\sigma_3$"]):
    v = np.zeros(3); v[i] = 1.9
    ax.plot([0, v[0]], [0, v[1]], [0, v[2]], color="0.4", lw=0.8)
    ax.text(*(1.08 * v), lab, fontsize=9)
ax.set_axis_off()
ax.set_xlim(-0.8, 1.9); ax.set_ylim(-0.8, 1.9); ax.set_zlim(-0.8, 1.9)
ax.set_box_aspect((1, 1, 1), zoom=1.15)
ax.view_init(elev=18, azim=25)
ax.set_title("(a) von Mises cylinder", fontsize=9)

# --- (b) pi-plane: von Mises circle and Tresca hexagon ----------------------
ax = fig.add_subplot(1, 3, 2)
ax.plot(R * np.cos(th), R * np.sin(th), color=blue, lw=2, label="von Mises")
# Tresca: vertices at uniaxial states (radius R), on the projected axes
ang = np.pi / 2 + np.arange(7) * np.pi / 3
ax.plot(R * np.cos(ang), R * np.sin(ang), color=orange, lw=2, label="Tresca")
for k, lab in enumerate([r"$\sigma_1$", r"$\sigma_2$", r"$\sigma_3$"]):
    a = np.pi / 2 + k * 2 * np.pi / 3
    ax.annotate("", xy=(1.1 * np.cos(a), 1.1 * np.sin(a)), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="0.4", lw=0.8))
    ax.text(1.2 * np.cos(a), 1.2 * np.sin(a), lab, ha="center", va="center")
ax.text(R / 2, 0.05, r"$\sqrt{2/3}\,\sigma_Y$", fontsize=8, color=blue, ha="center")
ax.plot([0, R], [0, 0], color=blue, lw=0.8)
ax.set_aspect("equal"); ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.25, 1.35)
ax.set_axis_off(); ax.legend(loc="lower left", fontsize=8, frameon=False)
ax.set_title(r"(b) deviatoric ($\pi$) plane", fontsize=9)

# --- (c) tension-torsion plane ---------------------------------------------
ax = fig.add_subplot(1, 3, 3)
ax.plot(np.cos(th), np.sin(th) / np.sqrt(3), color=blue, lw=2,
        label=r"von Mises $\sigma^2+3\tau^2=\sigma_Y^2$")
ax.plot(np.cos(th), np.sin(th) / 2, color=orange, lw=2,
        label=r"Tresca $\sigma^2+4\tau^2=\sigma_Y^2$")
ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
ax.plot([0], [1 / np.sqrt(3)], "o", color=blue, ms=4)
ax.plot([0], [0.5], "o", color=orange, ms=4)
ax.text(0.05, 1 / np.sqrt(3) + 0.03, r"$\sigma_Y/\sqrt{3}$", fontsize=8, color=blue)
ax.text(0.05, 0.38, r"$\sigma_Y/2$", fontsize=8, color=orange)
ax.set_xlabel(r"$\sigma/\sigma_Y$"); ax.set_ylabel(r"$\tau/\sigma_Y$")
ax.set_aspect("equal"); ax.set_ylim(-0.7, 0.8)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=7, frameon=False, ncol=1)
ax.grid(alpha=0.25)
ax.set_title("(c) tension\u2013torsion", fontsize=9)

fig.tight_layout()
fig.savefig("../../../figures/ch4/yield_loci.pdf")
