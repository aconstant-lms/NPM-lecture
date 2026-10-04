"""
Generates figures/ch4/hardening_surfaces.pdf (Chapter 4, Section 4.3.5).
Evolution of the von Mises yield surface in the tension-torsion plane
(sigma, sqrt(3) tau) after tensile loading to the point A: isotropic hardening
(expansion) and kinematic hardening (translation). Stresses in units of sigma_Y.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

blue, orange = "#1F5AC8", "#D9822B"
th = np.linspace(0, 2 * np.pi, 200)
A = 1.5                                   # tensile stress reached, sigma_A/sigma_Y
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
for ax, kind in zip(axes, ["isotropic", "kinematic"]):
    ax.plot(np.cos(th), np.sin(th), color="0.55", lw=1.2, ls="--",
            label="initial surface")
    # isotropic: circle of radius A centred at 0; kinematic: radius 1, centre A - 1
    if kind == "isotropic":
        c, r = 0.0, A
    else:
        c, r = A - 1.0, 1.0
    ax.plot(c + r * np.cos(th), r * np.sin(th), color=blue, lw=2,
            label="after loading to A")
    ax.plot([c], [0], "+", color=blue, ms=8)
    ax.annotate("", xy=(A, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=orange, lw=1.6))
    ax.plot([A], [0], "o", color=orange, ms=5)
    ax.text(A + 0.06, 0.08, "A", color=orange)
    ax.arrow(A, 0, 0.5, 0, color="k", width=0.012, head_width=0.09,
             length_includes_head=True)
    ax.text(A + 0.22, -0.3, "plastic\nflow", fontsize=7)
    rev = c - r                                     # reverse yield in tension-compression
    ax.plot([rev], [0], "s", color=blue, ms=4)
    ax.text(rev - 0.05, 0.1, "reverse\nyield", fontsize=7, ha="right", color=blue)
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    ax.set_aspect("equal"); ax.set_xlim(-2.0, 2.3); ax.set_ylim(-1.7, 1.7)
    ax.set_xlabel(r"$\sigma/\sigma_Y$"); ax.set_ylabel(r"$\sqrt{3}\,\tau/\sigma_Y$")
    ax.set_title(f"{kind} hardening", fontsize=9)
    ax.grid(alpha=0.25)
axes[0].legend(loc="lower left", fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig("../../../figures/ch4/hardening_surfaces.pdf")
