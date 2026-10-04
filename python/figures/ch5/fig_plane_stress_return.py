"""
Generates figures/ch5/plane_stress_return.pdf (Chapter 5, Section 5.7).
Projected return of Simo and Taylor (Box 5.5), perfect von Mises plasticity,
principal plane stresses (sigma_12 = 0), stresses in units of sY. The yield
locus is the ellipse s11^2 - s11 s22 + s22^2 = sY^2. The return goes from the
trial stress along C P sigma_{n+1}: it is neither radial (towards the origin)
nor along the Euclidean normal P sigma_{n+1}. The dotted curve is the family
(I + dg C P)^{-1} sigma_trial, dg >= 0, on which the solution is sought.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "../../examples/ch5")
from ci_core import plane_stress_return, Linear

blue, orange, green = "#1F5AC8", "#D9822B", "#14963C"
# Plane-stress moduli C and von Mises matrix P of Section 5.7 (vectors [s11, s22, s12])
E, nu, sY = 200e3, 0.3, 250.0
mu = E / (2 * (1 + nu))
C = E / (1 - nu**2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
P = np.array([[2, -1, 0], [-1, 2, 0], [0, 0, 6]]) / 3.0

# Box 5.5 with a uniaxial trial stress 2.6 sY, perfect plasticity
s_tr = np.array([2.6, 0.0, 0.0]) * sY                 # trial stress: uniaxial
eps = np.linalg.solve(C, s_tr)                        # strain giving this trial
s, ep, al, dg = plane_stress_return(eps, np.zeros(3), 0.0, E, nu, Linear(sY, 0.0))
# Radial point: the trial stress scaled onto the ellipse (not the solution)
lam = sY / np.sqrt(1.5 * s_tr @ P @ s_tr)             # radial point
s_rad = lam * s_tr
print(f"trial {s_tr[:2] / sY}, return {s[:2] / sY}, radial {s_rad[:2] / sY}, dg = {dg:.3e}")

# Yield ellipse s11^2 - s11 s22 + s22^2 = sY^2, in units of sY
th = np.linspace(0, 2 * np.pi, 400)                   # ellipse, principal axes at 45 deg
u = np.sqrt(2.0) * np.cos(th)                         # (s11 + s22)/sqrt2 semi-axis sqrt2
v = np.sqrt(2.0 / 3.0) * np.sin(th)                   # (s11 - s22)/sqrt2 semi-axis sqrt(2/3)
x, y = (u + v) / np.sqrt(2), (u - v) / np.sqrt(2)

fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3), gridspec_kw=dict(width_ratios=[1.25, 1]))
# Dotted curve (I + dg C P)^-1 sigma_trial for dg >= 0, through the modes A, B of
# Box 5.5; nrm is the Euclidean normal P sigma_{n+1}
d = np.concatenate([np.linspace(0, 5 * dg, 300), np.linspace(5 * dg, 400 * dg, 300)])
A = 1 + E * d / (3 * (1 - nu)); B = 1 + 2 * mu * d
S, D = (s_tr[0] + s_tr[1]) / A, (s_tr[0] - s_tr[1]) / B
nrm = P @ s; nrm = nrm[:2] / np.linalg.norm(nrm[:2])
T, Sn, Sr = s_tr[:2] / sY, s[:2] / sY, s_rad[:2] / sY
for ax, zoom in zip(axes, [False, True]):
    ax.fill(x, y, color=blue, alpha=0.08, lw=0)
    ax.plot(x, y, color=blue, lw=1.8)
    ax.axhline(0, color="0.6", lw=0.6); ax.axvline(0, color="0.6", lw=0.6)
    ax.plot(0.5 * (S + D) / sY, 0.5 * (S - D) / sY, ":", color="0.3", lw=1.3)
    ax.plot([0, T[0]], [0, T[1]], "--", color="0.55", lw=1.0)
    ax.annotate("", xy=Sn, xytext=T,
                arrowprops=dict(arrowstyle="-|>", color=orange, lw=2.0, shrinkA=3, shrinkB=4))
    ax.plot(*T, "o", color=orange, ms=6)
    ax.plot(*Sn, "o", color=blue, ms=6)
    ax.plot(*Sr, "o", color="0.45", ms=5, mfc="white")
    ax.set_xlabel(r"$\sigma_{11}/\sigma_Y$")
    ax.set_aspect("equal")
ax = axes[0]
ax.set_ylabel(r"$\sigma_{22}/\sigma_Y$")
ax.set_xlim(-1.3, 2.9); ax.set_ylim(-1.3, 1.3)
ax.text(T[0] - 0.1, T[1] - 0.22, r"$\sigma^{\mathrm{trial}}$", color=orange, fontsize=11)
ax.text(-1.0, 0.95, r"$f\leq 0$", color=blue, fontsize=10)
x0, x1, y0, y1 = 0.75, 1.55, -0.2, 0.45
ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="0.3", lw=0.7)
ax.text(x0, y1 + 0.06, "zoom", fontsize=8, color="0.3")
ax.set_title("(a) principal plane stresses", fontsize=10)
ax = axes[1]
ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
ax.annotate("", xy=Sn + 0.22 * nrm, xytext=Sn,
            arrowprops=dict(arrowstyle="-|>", color=green, lw=1.3))
ax.text(Sn[0] + 0.22 * nrm[0] + 0.01, Sn[1] + 0.22 * nrm[1] - 0.04, r"normal $P\sigma_{n+1}$",
        color=green, fontsize=8)
ax.text(Sn[0] - 0.2, Sn[1] - 0.075, r"$\sigma_{n+1}$", color=blue, fontsize=11)
ax.text(Sr[0] + 0.02, Sr[1] - 0.05, "radial point", color="0.35", fontsize=8)
ax.text(1.2, 0.175, r"return along $CP\sigma_{n+1}$", color=orange, fontsize=8, rotation=-5)
ax.text(0.765, 0.175, r"$(I+\Delta\gamma\,CP)^{-1}\sigma^{\mathrm{trial}}$",
        color="0.3", fontsize=8)
ax.set_title("(b) zoom on the return", fontsize=10)
fig.tight_layout()
fig.savefig("../../../figures/ch5/plane_stress_return.pdf")
