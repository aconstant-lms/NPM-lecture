"""
Generates figures/ch4/domain_evolution.pdf (Chapter 4, Section 4.1).
Movement of the elastic domain of a von Mises material along the
tension-torsion path O -> A (tension) -> B (torsion at constant tension),
in the plane (sigma, sqrt(3) tau), in units of sigma_Y. Panels (a)-(c):
isotropic, kinematic (Prager) and combined hardening, computed by a
stress-driven update of the centre and radius of the circle. Panel (d):
the distorted surface reported by Bui's tension-torsion experiments
(corner at the loading point, flattened flanks, rear side almost unchanged),
drawn schematically as the convex hull of the initial circle and of A.
The plastic strain increments (black arrows) are normal to the current
surface (associative flow); at the corner they lie in the cone of normals.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# Figure 4.2. The von Mises surface is the circle |x - c| = R in the plane
# (sigma, sqrt3 tau); initial surface: c = 0, R = 1 (units of sY).
blue, green, orange, grey = "#1F5AC8", "#14963C", "#D9822B", "0.55"
th = np.linspace(0, 2 * np.pi, 400)
A = np.array([1.3, 0.0])
B = np.array([1.3, 1.0])


def run(beta, n=4000):
    """Stress-driven path O->A->B. beta = isotropic fraction of hardening."""
    c, R = np.zeros(2), 1.0
    path = np.vstack([np.linspace([0, 0], A, n), np.linspace(A, B, n)[1:]])
    snaps, normals = {}, []
    for k, x in enumerate(path):
        d = x - c
        r = np.linalg.norm(d)
        # outside the current circle: the overshoot e is taken by hardening, a fraction
        # beta by expansion (R) and 1 - beta by translation of the centre c along the
        # normal (Prager's rule)
        if r > R:
            e, nrm = r - R, d / r
            R += beta * e
            c = c + (1 - beta) * e * nrm
            # store the loading point and the normal at A, half-way to B, and at B
            if k in (n - 1, n - 1 + n // 2, 2 * n - 2):
                normals.append((x.copy(), nrm.copy()))
        if k == n - 1:
            snaps["A"] = (c.copy(), R)
    snaps["B"] = (c.copy(), R)
    return snaps, normals


def frame(ax, title):
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.plot(np.cos(th), np.sin(th), color=grey, lw=1.2, ls="--")
    ax.set_aspect("equal")
    ax.set_xlim(-1.85, 2.6)
    ax.set_ylim(-1.75, 2.05)
    ax.set_xlabel(r"$\sigma/\sigma_Y$", fontsize=9)
    ax.set_ylabel(r"$\sqrt{3}\,\tau/\sigma_Y$", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.2)
    ax.set_title(title, fontsize=9)


def path_and_labels(ax):
    ax.plot([0, A[0], B[0]], [0, A[1], B[1]], color=orange, lw=1.8)
    ax.annotate("", xy=(0.75, 0), xytext=(0.45, 0),
                arrowprops=dict(arrowstyle="->", color=orange, lw=1.6))
    ax.annotate("", xy=(A[0], 0.6), xytext=(A[0], 0.35),
                arrowprops=dict(arrowstyle="->", color=orange, lw=1.6))
    for P, lab, off in ((A, "A", (0.05, -0.22)), (B, "B", (-0.22, 0.05))):
        ax.plot(*P, "o", color=orange, ms=4.5, zorder=5)
        ax.text(P[0] + off[0], P[1] + off[1], lab, color=orange, fontsize=9)
    ax.text(-0.17, -0.2, "O", fontsize=9)


def arrow(ax, x, nrm, L=0.55):
    ax.annotate("", xy=x + L * nrm, xytext=x,
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.3,
                                mutation_scale=10), zorder=6)


fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.3))
cases = [(1.0, "(a) isotropic hardening"), (0.0, "(b) kinematic hardening"),
         (0.5, "(c) combined hardening")]
for ax, (beta, title) in zip(axes.flat[:3], cases):
    frame(ax, title)
    snaps, normals = run(beta)
    for key, col, ls in (("A", blue, ":"), ("B", blue, "-")):
        c, R = snaps[key]
        ax.plot(c[0] + R * np.cos(th), c[1] + R * np.sin(th), color=col,
                lw=1.3 if key == "A" else 2.0, ls=ls)
        ax.plot(*c, "+", color=col, ms=7)
    path_and_labels(ax)
    for x, nrm in normals:
        arrow(ax, x, nrm)
axes[0, 0].plot([], [], color=grey, ls="--", label="initial surface")
axes[0, 0].plot([], [], color=blue, ls=":", label="surface at A")
axes[0, 0].plot([], [], color=blue, ls="-", lw=2, label="surface at B")
axes[0, 0].plot([], [], color="k", marker=r"$\rightarrow$", ls="none", ms=12,
                label=r"$\mathrm{d}\varepsilon^p$ (normal)")
axes[0, 0].legend(loc="lower left", fontsize=7, frameon=True, framealpha=0.9)

# (d) distortion observed in tension-torsion tests (schematic)
ax = axes[1, 1]
frame(ax, "(d) distortion observed (schematic, after Bui)")
c0 = np.array([0.12, 0.0])               # slight translation of the rear part
pts = np.column_stack([c0[0] + np.cos(th), c0[1] + np.sin(th)])
# convex hull of the translated circle and of the loading point A:
# the tangent points from A to the circle
d = np.linalg.norm(A - c0)
phi = np.arccos(1.0 / d)                 # half-angle of the arc kept
ang = np.arctan2(*(A - c0)[::-1])
keep = np.abs((th - ang + np.pi) % (2 * np.pi) - np.pi) >= phi
arc = pts[keep]
# order the kept arc starting after the upper tangent point
t_up = ang + phi
order = np.argsort((th[keep] - t_up) % (2 * np.pi))
hull = np.vstack([A, arc[order], A])
ax.add_patch(Polygon(hull, closed=True, fc=blue, alpha=0.08, ec="none"))
ax.plot(hull[:, 0], hull[:, 1], color=blue, lw=2)
# normal cone at the corner A
n1 = np.array([np.cos(ang + phi), np.sin(ang + phi)])
n2 = np.array([np.cos(ang - phi), np.sin(ang - phi)])
cone = np.vstack([A, A + 0.75 * n1, A + 0.75 * n2])
ax.add_patch(Polygon(cone, closed=True, fc=orange, alpha=0.25, ec="none"))
ax.plot(*np.column_stack([A, A + 0.75 * n1]), color=orange, lw=0.9)
ax.plot(*np.column_stack([A, A + 0.75 * n2]), color=orange, lw=0.9)
arrow(ax, A, np.array([1.0, 0.0]), L=0.6)
# smooth points on the flank and at the rear: normal flow
for t in (ang + phi + 0.6, np.pi):
    x = c0 + np.array([np.cos(t), np.sin(t)])
    arrow(ax, x, np.array([np.cos(t), np.sin(t)]), L=0.45)
ax.plot([0, A[0]], [0, 0], color=orange, lw=1.8)
ax.plot(*A, "o", color=orange, ms=4.5, zorder=5)
ax.text(A[0] + 0.05, -0.25, "A", color=orange, fontsize=9)
ax.text(1.55, 0.95, "corner,\nflat flanks", fontsize=7.5, color=blue)
ax.text(-1.8, 1.25, "rear side\nalmost fixed", fontsize=7.5, color=blue)
ax.text(1.95, -0.62, "cone of\nnormals", fontsize=7.5, color=orange)

fig.tight_layout()
fig.savefig("../../../figures/ch4/domain_evolution.pdf")
