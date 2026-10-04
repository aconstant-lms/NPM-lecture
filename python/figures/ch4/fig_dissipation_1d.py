"""
Generates figures/ch4/dissipation_1d.pdf (Chapter 4, Section 4.4.2).
The dissipation potential of the perfectly plastic filament,
phi(epsdot_p) = sigma_Y |epsdot_p|, and its Legendre-Fenchel conjugate, the
indicator function of the elastic domain [-sigma_Y, sigma_Y]. The slopes of
phi (subdifferential, [-sigma_Y, sigma_Y] at the origin) are the points of
the domain of phi*; the normal cone of [-sigma_Y, sigma_Y] at +-sigma_Y is
the set of admissible rates.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Figure 4.13; sigma_Y = 1 in both panels (schematic, no numbers)
blue, green, orange = "#1F5AC8", "#14963C", "#D9822B"
fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9))

# (a) phi(d) = sigma_Y |d|, sigma_Y = 1
ax = axes[0]
d = np.linspace(-1.6, 1.6, 3)
ax.plot([-1.6, 0, 1.6], [1.6, 0, 1.6], color=blue, lw=2.2)
for s in (-0.55, 0.3):                     # supporting lines at the kink
    ax.plot(d, s * d, color=orange, lw=0.9, ls="--")
ax.text(1.0, 1.45, r"slope $\sigma_Y$", fontsize=8, color=blue, ha="right")
ax.text(-1.0, 1.45, r"slope $-\sigma_Y$", fontsize=8, color=blue, ha="left")
ax.text(1.6, -0.22, "supporting lines at 0:\n" r"slopes $\sigma\in[-\sigma_Y,\sigma_Y]$",
        fontsize=8, color=orange, ha="center", va="center")
ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
ax.set_xlim(-2.5, 2.5); ax.set_ylim(-1.0, 2.0)
ax.set_xlabel(r"$\dot\varepsilon^p$")
ax.set_ylabel(r"$\varphi(\dot\varepsilon^p)=\sigma_Y|\dot\varepsilon^p|$")
ax.set_xticks([]); ax.set_yticks([])
ax.set_title(r"(a) dissipation potential $\varphi$", fontsize=9)

# (b) phi*(sigma) = indicator of [-sigma_Y, sigma_Y]
ax = axes[1]
ax.fill_between([-1, 1], 0, 0.06, color=green, alpha=0.3, lw=0)
ax.plot([-1, 1], [0, 0], color=blue, lw=3)
for x in (-1, 1):
    ax.plot([x, x], [0, 1.75], color=blue, lw=2.2)
    ax.annotate("", xy=(x, 2.0), xytext=(x, 1.75),
                arrowprops=dict(arrowstyle="->", color=blue, lw=1.5))
    ax.plot([x], [0], "o", color="k", ms=4, zorder=5)
ax.text(1.1, 1.75, r"$+\infty$", fontsize=9, color=blue)
ax.text(-1.1, 1.75, r"$+\infty$", fontsize=9, color=blue, ha="right")
ax.text(0, 0.15, r"$\varphi^*=0$ on $\mathbb{E}_\sigma$", fontsize=8, ha="center",
        color=blue)
ax.text(-1, -0.28, r"$-\sigma_Y$", ha="center", fontsize=9)
ax.text(1, -0.28, r"$\sigma_Y$", ha="center", fontsize=9)
# normal cones at the end points
ax.annotate("", xy=(1.7, 0.0), xytext=(1.0, 0.0),
            arrowprops=dict(arrowstyle="-|>", color=orange, lw=1.5))
ax.annotate("", xy=(-1.7, 0.0), xytext=(-1.0, 0.0),
            arrowprops=dict(arrowstyle="-|>", color=orange, lw=1.5))
ax.text(1.85, 0.55, "normal cone\n" r"$\dot\varepsilon^p\geq0$", fontsize=8,
        color=orange, ha="center")
ax.text(-1.85, 0.55, "normal cone\n" r"$\dot\varepsilon^p\leq0$", fontsize=8,
        color=orange, ha="center")
ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
ax.set_xlim(-2.5, 2.5); ax.set_ylim(-1.0, 2.0)
ax.set_xlabel(r"$\sigma$")
ax.set_ylabel(r"$\varphi^*(\sigma)=I_{[-\sigma_Y,\sigma_Y]}(\sigma)$")
ax.set_xticks([]); ax.set_yticks([])
ax.set_title(r"(b) its conjugate $\varphi^*$: indicator of $\mathbb{E}_\sigma$",
             fontsize=9)

fig.tight_layout()
fig.savefig("../../../figures/ch4/dissipation_1d.pdf")
