"""
Generates figures/ch2/kirsch.pdf (Chapter 2, exercise "A plate with a hole:
Kirsch against finite elements"). Quarter of a square plate (side 2L = 40a)
with a hole of radius a under remote tension s along x, linear triangles
(python/examples/ch2/kirsch_fem.py, 48 cells along the hole):
(a) the mesh near the hole, (b) hoop stress on the hole, (c) sigma_xx on the
ligament x = 0, FEM against the Kirsch solution of the infinite plate.
"""
import sys
import numpy as np
from style_ch2 import plt, BLUE, ORANGE, GRAY, OUT
sys.path.insert(0, "../../examples/ch2")
from kirsch_fem import solve, mesh, kirsch

nt = 48
X, sig, ndof = solve(nt)
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5),
                         gridspec_kw=dict(width_ratios=[0.8, 1, 1]))
# (a) mesh near the hole (a coarser mesh for legibility)
ax = axes[0]
Xc, Tc = mesh(16, 1.0, 20.0)
ax.triplot(Xc[:, 0], Xc[:, 1], Tc, color=GRAY, lw=0.35)
ax.set_xlim(0, 4); ax.set_ylim(0, 4); ax.set_aspect("equal")
ax.set_xticks([0, 1, 2, 3, 4]); ax.set_yticks([0, 1, 2, 3, 4])
ax.set_xlabel(r"$x_1/a$"); ax.set_ylabel(r"$x_2/a$")
ax.set_title("(a) mesh near the hole")

# (b) hoop stress on the hole r = a
ax = axes[1]
hole = np.arange(nt + 1)                   # nodes i = 0
th = np.arctan2(X[hole, 1], X[hole, 0])
c, s = np.cos(th), np.sin(th)
stt = sig[hole, 0] * s**2 + sig[hole, 1] * c**2 - 2 * sig[hole, 2] * s * c
tt = np.linspace(0, np.pi / 2, 200)
ax.plot(np.degrees(tt), kirsch(1.0, tt)[1], color="k", lw=1.3, label="Kirsch")
ax.plot(np.degrees(th), stt, "o", color=BLUE, ms=3, mfc="none", label="finite elements")
ax.axhline(0, color=GRAY, lw=0.5)
ax.set_xlabel(r"$\theta$ (degrees)"); ax.set_ylabel(r"$\sigma_{\theta\theta}/\sigma$ on $r=a$")
ax.set_xticks([0, 30, 60, 90]); ax.set_ylim(-1.2, 3.3)
ax.legend(frameon=False, loc="upper left")
ax.set_title(r"(b) hoop stress on the hole")

# (c) sigma_xx along the ligament x = 0
ax = axes[2]
lig = np.where(np.isclose(X[:, 0], 0) & (X[:, 1] <= 6))[0]
y = np.linspace(1, 6, 200)
ax.plot(y, 1 + 0.5 / y**2 + 1.5 / y**4, color="k", lw=1.3)
ax.plot(X[lig, 1], sig[lig, 0], "o", color=BLUE, ms=3, mfc="none")
ax.axhline(1, color=GRAY, lw=0.5, ls="--")
ax.set_xlabel(r"$x_2/a$ on $x_1=0$"); ax.set_ylabel(r"$\sigma_{11}/\sigma$")
ax.set_xlim(1, 6); ax.set_ylim(0.8, 3.3)
ax.set_title(r"(c) stress on the ligament")
fig.tight_layout()
fig.savefig(OUT + "kirsch.pdf")
