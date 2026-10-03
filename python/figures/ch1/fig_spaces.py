"""
Generates figures/ch1/spaces.pdf (Chapter 1, Section "Function spaces in a
nutshell", Table "Examples and counterexamples of finite-energy functions").
(a) hat function and its derivative (a step): H^1, not H^2;
(b) step function, whose derivative is a Dirac mass: L^2, not H^1;
(c) x^alpha on (0,1): in H^1 iff alpha > 1/2;
(d) ln|ln r| (H^1 in 2D, unbounded) against ln r (not H^1);
(e) crack-tip field u = r^(1/2) cos(theta/2), crack along theta = +-pi;
(f) its level lines; |grad u| = 1/(2 sqrt r) is singular but
    |grad u|^2 = 1/(4r) is integrable in 2D (r dr dtheta).
Run: cd python/figures/ch1 && python3 fig_spaces.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

BLUE, GREEN, ORANGE, GRAY = "#1F5AC8", "#14963C", "#D85A30", "#666666"
fig = plt.figure(figsize=(10, 6.6))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.35])
axs = [fig.add_subplot(gs[0, k]) for k in range(4)]


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7)


# (a) hat
x = np.linspace(-1.5, 1.5, 601)
ax = axs[0]
ax.plot(x, np.clip(1 - np.abs(x), 0, None), color=BLUE, lw=1.5, label="$v$")
dv = np.where(np.abs(x) < 1, -np.sign(x), 0.0)
ax.plot(x, dv, color=ORANGE, lw=1.2, ls="--", label="$v'$")
ax.set_title("(a) hat: $H^1$, not $H^2$", fontsize=9)
ax.legend(frameon=False, fontsize=7, loc="lower left")
clean(ax)

# (b) step
ax = axs[1]
ax.plot([-1.5, 0], [0, 0], color=BLUE, lw=1.5)
ax.plot([0, 1.5], [1, 1], color=BLUE, lw=1.5, label="$v$")
ax.annotate("", xy=(0, 1.35), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.4))
ax.text(0.08, 1.25, r"$v'=\delta_0$", color=ORANGE, fontsize=8)
ax.set_ylim(-0.3, 1.5)
ax.set_title("(b) step: $L^2$, not $H^1$", fontsize=9)
clean(ax)

# (c) x^alpha
ax = axs[2]
x = np.linspace(0, 1, 1001)
for a, c in zip([0.25, 0.5, 0.75, 1.0], [ORANGE, "#B0603A", GREEN, BLUE]):
    ax.plot(x, x**a, color=c, lw=1.3, label=rf"$\alpha={a:g}$")
ax.set_title(r"(c) $x^\alpha\in H^1$ iff $\alpha>1/2$", fontsize=9)
ax.legend(frameon=False, fontsize=7, loc="lower right")
ax.set_xlabel("$x$", fontsize=8, labelpad=0)
clean(ax)

# (d) ln|ln r| against ln r
ax = axs[3]
r = np.logspace(-12, np.log10(0.5), 400)
ax.semilogx(r, np.log(np.abs(np.log(r))), color=BLUE, lw=1.5,
            label=r"$\ln|\ln r|$: $H^1$")
ax.semilogx(r, -np.log(r) / 5, color=ORANGE, lw=1.2, ls="--",
            label=r"$-\frac{1}{5}\ln r$: not $H^1$")
ax.set_xlabel("$r$", fontsize=8, labelpad=0)
ax.set_title(r"(d) unbounded, yet $H^1$ (2D)", fontsize=9)
ax.legend(frameon=False, fontsize=7, loc="upper right")
clean(ax)

# (e) crack-tip field, surface
ax = fig.add_subplot(gs[1, :2], projection="3d")
R, T = np.meshgrid(np.linspace(0, 1, 60), np.linspace(-np.pi, np.pi, 121))
X, Y, U = R * np.cos(T), R * np.sin(T), np.sqrt(R) * np.cos(T / 2)
ax.plot_surface(X, Y, U, cmap=cm.viridis, rstride=4, cstride=4, lw=0.25,
                edgecolor=(0, 0, 0, 0.25))
ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1]); ax.set_zticks([0, 0.5, 1])
ax.plot([-1, 0], [0, 0], [0, 0], color=ORANGE, lw=2.5, zorder=10)
ax.set_xlabel("$x_1$", fontsize=8, labelpad=-6)
ax.set_ylabel("$x_2$", fontsize=8, labelpad=-6)
ax.tick_params(labelsize=6, pad=-2)
ax.view_init(elev=28, azim=-60)
ax.set_title(r"(e) crack-tip field $u=r^{1/2}\cos(\theta/2)$; crack in orange",
             fontsize=9)

# (f) level lines
ax = fig.add_subplot(gs[1, 2:])
cs = ax.contourf(X, Y, U, levels=14, cmap=cm.viridis)
ax.contour(X, Y, U, levels=14, colors="k", linewidths=0.3)
ax.plot([-1, 0], [0, 0], color=ORANGE, lw=2.5)
ax.set_aspect("equal")
ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
ax.tick_params(labelsize=7)
fig.colorbar(cs, ax=ax, shrink=0.8, pad=0.03).ax.tick_params(labelsize=7)
ax.set_title(r"(f) level lines: $|\nabla u|=\frac{1}{2\sqrt{r}}$,"
             r" $\int|\nabla u|^2<\infty$", fontsize=9)

fig.tight_layout()
fig.savefig("../../../figures/ch1/spaces.pdf")
# energy of the crack-tip field in the disc of radius rho: int |grad u|^2 = pi*rho/2
rho = 1.0
rr = np.linspace(1e-8, rho, 200001)
print("int_B |grad u|^2 =", 2 * np.pi * np.trapezoid(1 / (4 * rr) * rr, rr),
      "(exact pi/2 =", np.pi / 2, ")")
print("saved figures/ch1/spaces.pdf")
