"""
Generates figures/ch2/reference_map.pdf (Chapter 2, paragraph "Numerical
integration with Gauss points"). Bilinear map x(xi) = sum_a N_a(xi) x_a from
the reference square [-1,1]^2 to a quadrilateral element: coordinate lines,
the 2 x 2 Gauss points and, at one point, the columns of the Jacobian matrix
J = dx/dxi (tangent vectors to the coordinate lines).
"""
import numpy as np
from matplotlib.patches import FancyArrowPatch
from style_ch2 import plt, BLUE, ORANGE, GREEN, GRAY, OUT

Xa = np.array([[0.0, 0.0], [2.4, 0.35], [2.75, 2.1], [0.35, 1.7]])  # nodes 1..4
xa = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float)

def N(xi, eta):
    return 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                            (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])

def x_of(xi, eta):
    return np.tensordot(N(xi, eta), Xa, axes=(0, 0))

def jac(xi, eta):
    dN = 0.25 * np.array([[-(1 - eta), -(1 - xi)], [(1 - eta), -(1 + xi)],
                          [(1 + eta), (1 + xi)], [-(1 + eta), (1 - xi)]])
    return Xa.T @ dN                       # J[i, k] = d x_i / d xi_k

fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8),
                         gridspec_kw=dict(width_ratios=[1, 1.25]))
t = np.linspace(-1, 1, 50)
g = 1 / np.sqrt(3)
gp = [(-g, -g), (g, -g), (g, g), (-g, g)]
p0 = (0.35, -0.3)                          # point where J is drawn
for ax, mapped in [(axes[0], False), (axes[1], True)]:
    f = (lambda a, b: x_of(a, b).T) if mapped else (lambda a, b: np.array([a + 0 * b, b + 0 * a]))
    for c in np.linspace(-1, 1, 9):
        lw, col = (1.3, "k") if abs(c) == 1 else (0.5, GRAY)
        ax.plot(*f(t, c + 0 * t), color=col, lw=lw)
        ax.plot(*f(c + 0 * t, t), color=col, lw=lw)
    P = Xa if mapped else xa
    for k, (px, py) in enumerate(P):
        ax.plot(px, py, "o", color="k", ms=4)
        off = np.sign(np.array([px, py]) - P.mean(0)) * 0.13
        ax.text(px + off[0], py + off[1], str(k + 1), ha="center", va="center", fontsize=8)
    for a, b in gp:
        ax.plot(*f(np.array([a]), np.array([b])), "x", color=ORANGE, ms=6, mew=1.5)
    q = f(np.array([p0[0]]), np.array([p0[1]])).ravel()
    J = jac(*p0) if mapped else np.eye(2)
    s = 0.45 if mapped else 0.55
    for k, (col, lab) in enumerate([(BLUE, r"$\partial\boldsymbol{x}/\partial\xi$" if mapped else r"$\xi$"),
                                     (GREEN, r"$\partial\boldsymbol{x}/\partial\eta$" if mapped else r"$\eta$")]):
        e = J[:, k] * s
        ax.add_patch(FancyArrowPatch(q, q + e, arrowstyle="-|>", mutation_scale=9,
                                     color=col, lw=1.4, zorder=5))
        ax.text(*(q + e * 1.15 + np.array([0.03, 0.03])), lab, color=col, fontsize=8.5,
                bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.9))
    ax.set_aspect("equal"); ax.axis("off")
axes[0].set_title(r"reference element $\hat\Omega=[-1,1]^2$")
axes[1].set_title(r"element $\Omega_e$")
axes[0].set_xlim(-1.35, 1.45); axes[0].set_ylim(-1.35, 1.35)
fig.tight_layout(w_pad=3)
fig.add_artist(FancyArrowPatch((0.425, 0.55), (0.52, 0.55), transform=fig.transFigure,
                               arrowstyle="-|>", mutation_scale=12, color="k", lw=1.2,
                               connectionstyle="arc3,rad=-0.3"))
fig.text(0.472, 0.66, r"$\boldsymbol{x}(\boldsymbol{\xi})$", ha="center", fontsize=9)
fig.text(0.472, 0.36, r"$\mathbf{J}=\partial\boldsymbol{x}/\partial\boldsymbol{\xi}$",
         ha="center", fontsize=8)
fig.savefig(OUT + "reference_map.pdf")
print("det J at the Gauss points:", [round(np.linalg.det(jac(a, b)), 3) for a, b in gp])
