"""
Generates figures/ch4/dp_mc.pdf (Chapter 4, Section 4.3.2).
Pressure-sensitive criteria in the space of principal stresses (tension
positive): the Mohr-Coulomb pyramid
    (sigma_1 - sigma_3) + (sigma_1 + sigma_3) sin(phi) - 2 c cos(phi) <= 0,
    sigma_1 >= sigma_2 >= sigma_3,
and the Drucker-Prager cone circumscribed to it (same apex, through the
compression vertices). Friction angle phi = 30 deg, cohesion c = 1.
(c): the sections of both by the deviatoric plane through the origin.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

blue, orange, green = "#1F5AC8", "#D9822B", "#14963C"
phi, coh = np.deg2rad(30.0), 1.0
p_apex = coh / np.tan(phi)                      # apex on the hydrostatic axis
h = np.array([1, 1, 1]) / np.sqrt(3)            # hydrostatic axis
e1 = np.array([2, -1, -1]) / np.sqrt(6)         # basis of the pi-plane
e2 = np.array([0, 1, -1]) / np.sqrt(2)

om = np.linspace(0, 2 * np.pi, 361)
dirs = np.outer(np.cos(om), e1) + np.outer(np.sin(om), e2)   # unit deviators
srt = -np.sort(-dirs, axis=1)                                 # s1>=s2>=s3
a = (srt[:, 0] - srt[:, 2]) + (srt[:, 0] + srt[:, 2]) * np.sin(phi)


def rho_mc(p):
    """Radius of the Mohr-Coulomb section at mean stress p, per direction."""
    return 2 * np.sin(phi) * (p_apex - p) / a


def rho_dp(p):
    """Drucker-Prager cone through the compression vertices of MC."""
    return 2 * np.sin(phi) * (p_apex - p) / a.min() * np.ones_like(a)


def surface(ax, rho, color, title):
    """Draw the criterion in the frame (e1, e2, hydrostatic axis), so that
    the hydrostatic axis is vertical; coordinates (x, y, z) = (s.e1, s.e2,
    sqrt(3) p)."""
    P = np.linspace(p_apex, -2.0, 25)
    OM, PP = np.meshgrid(om, P)
    R = np.array([rho(p) for p in P])
    ax.plot_surface(R * np.cos(OM), R * np.sin(OM), np.sqrt(3) * PP,
                    color=color, alpha=0.25, linewidth=0, shade=True)
    for p, lw in ((P[-1], 1.0), (0.0, 1.4)):
        r = rho(p)
        ax.plot(r * np.cos(om), r * np.sin(om), np.sqrt(3) * p * np.ones_like(om),
                color=color, lw=lw)
    for k in range(0, 360, 60):            # meridians through the vertices
        ax.plot([0, rho(P[-1])[k] * np.cos(om[k])],
                [0, rho(P[-1])[k] * np.sin(om[k])],
                [np.sqrt(3) * p_apex, np.sqrt(3) * P[-1]], color=color, lw=0.6)
    ax.plot([0, 0], [0, 0], [np.sqrt(3) * P[-1] - 0.8, np.sqrt(3) * p_apex + 1.2],
            color="k", lw=1)
    ax.text(0.15, 0, np.sqrt(3) * p_apex + 1.25, "hydrostatic axis", fontsize=7)
    ax.scatter([0], [0], [np.sqrt(3) * p_apex], color="k", s=8)
    ax.text(0.25, 0, np.sqrt(3) * p_apex - 0.1, "apex", fontsize=7)
    for i, lab in enumerate([r"$\sigma_1$", r"$\sigma_2$", r"$\sigma_3$"]):
        v = 2.4 * np.array([e1[i], e2[i], h[i]])
        ax.plot([0, v[0]], [0, v[1]], [0, v[2]], color="0.35", lw=0.8)
        ax.text(*(1.1 * v), lab, fontsize=9)
    ax.set_axis_off()
    ax.set_xlim(-2.6, 2.6); ax.set_ylim(-2.6, 2.6)
    ax.set_zlim(np.sqrt(3) * -2.0 - 0.5, np.sqrt(3) * p_apex + 1.3)
    ax.set_box_aspect((1, 1, 1.25), zoom=1.15)
    ax.view_init(elev=18, azim=-35)
    ax.set_title(title, fontsize=9)


fig = plt.figure(figsize=(8.6, 3.3))
surface(fig.add_subplot(1, 3, 1, projection="3d"), rho_dp, blue,
        "(a) Drucker–Prager cone")
surface(fig.add_subplot(1, 3, 2, projection="3d"), rho_mc, orange,
        "(b) Mohr–Coulomb pyramid")

ax = fig.add_subplot(1, 3, 3)
for rho, col, lab in ((rho_dp, blue, "Drucker–Prager"),
                      (rho_mc, orange, "Mohr–Coulomb")):
    r = rho(0.0) * np.ones_like(om)
    # in-plane coordinates: sigma_1 axis drawn vertically
    x, y = -r * np.sin(om), r * np.cos(om)
    ax.plot(x, y, color=col, lw=2, label=lab)
for k, lab in enumerate([r"$\sigma_1$", r"$\sigma_2$", r"$\sigma_3$"]):
    ang = np.pi / 2 + k * 2 * np.pi / 3
    ax.annotate("", xy=(2.9 * np.cos(ang), 2.9 * np.sin(ang)), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="0.4", lw=0.8))
    ax.text(3.15 * np.cos(ang), 3.15 * np.sin(ang), lab, ha="center",
            va="center")
rt, rc = rho_mc(0.0).min(), rho_mc(0.0).max()
ax.text(0.12, rt * 0.55, "tension\nvertex", fontsize=7, color=orange)
ax.text(0.0, -rc - 0.35, "compression vertex", fontsize=7, color=orange,
        ha="center")
ax.set_aspect("equal"); ax.set_xlim(-3.3, 3.3); ax.set_ylim(-3.3, 3.4)
ax.set_axis_off()
ax.legend(loc="lower left", fontsize=7, frameon=False)
ax.set_title(r"(c) sections by the $\pi$-plane ($\phi=30^\circ$)", fontsize=9)

fig.tight_layout()
fig.savefig("../../../figures/ch4/dp_mc.pdf")
