"""
Generates figures/ch4/yield_loci.pdf (Chapter 4, Section 4.3.2).
Von Mises cylinder in the space of principal stresses, and sections of the
von Mises and Tresca criteria by the deviatoric (pi) plane, with the Lode
angle theta of a stress state (measured from the pure-shear direction,
-30 deg in uniaxial tension, +30 deg in uniaxial compression, sector
sigma_1 >= sigma_2 >= sigma_3). Stresses in units of sigma_Y.
The tension-torsion plane is in fig_classical_tests.py.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

blue, green, orange = "#1F5AC8", "#14963C", "#D9822B"
fig = plt.figure(figsize=(8.0, 3.9))

# --- (a) von Mises cylinder in principal stress space -----------------------
gs = fig.add_gridspec(1, 2, width_ratios=[0.8, 1.2])
ax = fig.add_subplot(gs[0], projection="3d")
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

# --- (b) pi-plane: von Mises circle, Tresca hexagon, Lode angle -------------
ax = fig.add_subplot(gs[1])
ax.plot(R * np.cos(th), R * np.sin(th), color=blue, lw=2, label="von Mises")
# Tresca: vertices at uniaxial states (radius R), on the projected axes
ang = np.pi / 2 + np.arange(7) * np.pi / 3
ax.plot(R * np.cos(ang), R * np.sin(ang), color=orange, lw=2, label="Tresca")
# sector sigma_1 >= sigma_2 >= sigma_3 (between +sigma_1 and -sigma_3)
sec = np.linspace(np.pi / 2, 5 * np.pi / 6, 30)
ax.fill(np.r_[0, 1.02 * np.cos(sec), 0], np.r_[0, 1.02 * np.sin(sec), 0],
        color=green, alpha=0.10, lw=0)
for k, lab in enumerate([r"$\sigma_1$", r"$\sigma_2$", r"$\sigma_3$"]):
    a = np.pi / 2 + k * 2 * np.pi / 3
    ax.annotate("", xy=(1.1 * np.cos(a), 1.1 * np.sin(a)), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="0.4", lw=0.8))
    ax.text(1.2 * np.cos(a), 1.2 * np.sin(a), lab, ha="center", va="center")
    ax.plot([0, -0.95 * np.cos(a)], [0, -0.95 * np.sin(a)], color="0.6",
            lw=0.6, ls=":")
# pure shear direction (theta = 0) and a stress state at theta = 15 deg
a0 = 2 * np.pi / 3
ax.plot([0, 1.05 * np.cos(a0)], [0, 1.05 * np.sin(a0)], color=green, lw=1,
        ls="--")
thL = np.deg2rad(15)
P = R * np.array([np.cos(a0 + thL), np.sin(a0 + thL)])
ax.annotate("", xy=P, xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color="k", lw=1.3))
ax.text(P[0] - 0.05, P[1] + 0.07, r"$\mathbfit{s}$", fontsize=11)
arc = np.linspace(a0, a0 + thL, 20)
ax.plot(0.42 * np.cos(arc), 0.42 * np.sin(arc), color="k", lw=1)
am = a0 + thL / 2
ax.text(0.53 * np.cos(am), 0.53 * np.sin(am), r"$\theta$", fontsize=11,
        ha="center", va="center")
# special states of the sector
for a, lab in ((np.pi / 2, "T"), (a0, "S"), (5 * np.pi / 6, "C")):
    ax.plot(R * np.cos(a), R * np.sin(a), "o", color=green, ms=4, zorder=5)
    ax.text(0.97 * np.cos(a) + (0.07 if lab == "T" else 0),
            0.97 * np.sin(a), lab, color=green, fontsize=9, ha="center",
            va="center")
ax.text(0.62, -1.05, "T: tension, $\\theta=-30^\\circ$, $\\mu=-1$\n"
        "S: shear, $\\theta=0$, $\\mu=0$\n"
        "C: compression, $\\theta=30^\\circ$, $\\mu=1$",
        fontsize=7.5, color=green, va="bottom")
ax.text(R / 2, -0.09, r"$\sqrt{2/3}\,\sigma_Y$", fontsize=8, color=blue,
        ha="center")
ax.plot([0, R], [0, 0], color=blue, lw=0.8)
ax.set_aspect("equal"); ax.set_xlim(-1.3, 1.75); ax.set_ylim(-1.25, 1.3)
ax.set_axis_off(); ax.legend(loc="lower left", fontsize=8, frameon=False)
ax.set_title(r"(b) deviatoric ($\pi$) plane and Lode angle", fontsize=9)

fig.tight_layout()
fig.savefig("../../../figures/ch4/yield_loci.pdf")
