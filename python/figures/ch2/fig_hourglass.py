"""
Generates figures/ch2/hourglass.pdf (Chapter 2, exercise "Gauss points",
part (c)). Bilinear square element (Q4) integrated with one Gauss point:
(a) the hourglass mode u = delta xi eta e_1 of one element, with zero strain
at the centre; (b) the same mode repeated with alternating signs over a
4 x 4 mesh: a global zero-energy mode.
Figure 2.15, Exercise 2.18 (c): with one Gauss point at the centre, the
hourglass mode has zero strain there, so it costs no energy.
Run: cd python/figures/ch2 && python3 fig_hourglass.py
"""
import numpy as np
from style_ch2 import plt, BLUE, ORANGE, GRAY, OUT

d = 0.28                                  # amplitude of the mode (exaggerated)

def q4_edges(Xn):
    """Closed polygon through the four nodes (counter-clockwise)."""
    return np.vstack([Xn, Xn[:1]])

fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9),
                         gridspec_kw=dict(width_ratios=[1, 1.35]))
# (a) one element
# Nodes of the reference square [-1, 1]^2; u_1 = delta xi eta is +-delta at the
# corners and its gradient vanishes at xi = eta = 0 (the one Gauss point).
ax = axes[0]
xi = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float)
u = d * xi[:, 0] * xi[:, 1]                # u_1 = delta xi eta at the nodes
Xd = xi + np.c_[u, 0 * u]
P = q4_edges(xi); ax.plot(P[:, 0], P[:, 1], "--", color=GRAY, lw=1)
P = q4_edges(Xd); ax.fill(P[:, 0], P[:, 1], color=BLUE, alpha=0.12, lw=0)
ax.plot(P[:, 0], P[:, 1], color=BLUE, lw=1.5)
for (x0, y0), du in zip(xi, u):
    ax.annotate("", xy=(x0 + du, y0), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))
    ax.plot(x0 + du, y0, "o", color=BLUE, ms=4)
for t in np.linspace(-1, 1, 5)[1:-1]:      # deformed horizontal fibres
    ax.plot([-1 - d * t, 1 + d * t], [t, t], color=BLUE, lw=0.5, alpha=0.6)
ax.plot(0, 0, "x", color="k", ms=7, mew=1.6)
ax.text(0.12, 0.1, "Gauss point:\n" + r"$\varepsilon=\mathbf{0}$", fontsize=7.5,
        bbox=dict(fc="white", ec="none", pad=1, alpha=0.85))
ax.text(-1.3, 1.25, r"$u_1=\delta\,\xi\eta$", fontsize=9, color=ORANGE)
ax.set_title("(a) hourglass mode of one element")
ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.45, 1.5)

# (b) mesh 4 x 4, nodal displacement u_1 = delta (-1)^(i+j)
ax = axes[1]
n, h = 4, 1.0
I, J = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
X0, Y0 = I * h, J * h
# Alternating nodal values: in every element the field is again the hourglass
# mode, so the strain vanishes at every element centre (crosses).
U = 0.5 * d * (-1.0) ** (I + J)
for X, Y, col, lw, ls in [(X0, Y0, GRAY, 0.8, "--"), (X0 + U, Y0, BLUE, 1.3, "-")]:
    for k in range(n + 1):
        ax.plot(X[k, :], Y[k, :], ls, color=col, lw=lw)
        ax.plot(X[:, k], Y[:, k], ls, color=col, lw=lw)
ax.plot((X0 + U).ravel(), Y0.ravel(), "o", color=BLUE, ms=3)
xc = (np.arange(n) + 0.5) * h
XC, YC = np.meshgrid(xc, xc)
ax.plot(XC.ravel(), YC.ravel(), "x", color="k", ms=5, mew=1.2)
ax.set_title("(b) the mode repeated over a mesh")
ax.set_xlim(-0.4, n + 0.4); ax.set_ylim(-0.3, n + 0.45)
for a in axes:
    a.set_aspect("equal"); a.axis("off")
fig.tight_layout()
fig.savefig(OUT + "hourglass.pdf")
