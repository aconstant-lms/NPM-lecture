"""
Generates figures/ch1/bolza.pdf (Figure 1.9; Chapter 1, Section 1.5 "Existence by
minimization: the direct method", paragraph on the Bolza problem).
Bolza problem: J(u) = int_0^1 ((u')^2 - 1)^2 + u^2 dx, u(0) = u(1) = 0.
The sawtooth u_n (n teeth, slopes +-1, height 1/(2n)) has J(u_n) = 1/(12 n^2):
J(u_n) -> 0 = inf J, u_n -> 0 uniformly, but J(0) = 1. The infimum is not
attained: the integrand is not convex in u', and J is not weakly lower
semicontinuous.
Run: cd python/figures/ch1 && python3 fig_bolza.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, GREEN, ORANGE, GRAY = "#1F5AC8", "#14963C", "#D85A30", "#666666"


def sawtooth(n, x):
    """n teeth of slope +-1 on (0,1), zero at x = k/n."""
    s = (x * n) % 1.0
    return np.minimum(s, 1 - s) / n


def J_sawtooth(n, x):
    """J(u_n) by the trapezoidal rule on the grid x; the double-well term
    ((u')^2 - 1)^2 vanishes since |u_n'| = 1, so J(u_n) = int u_n^2 = 1/(12 n^2)."""
    u = sawtooth(n, x)
    du = np.where((x * n) % 1.0 < 0.5, 1.0, -1.0)    # exact slope, a.e.
    return np.trapezoid((du**2 - 1)**2 + u**2, x)


# Fine grid (2e5 intervals) so that the quadrature resolves 64 teeth.
x = np.linspace(0, 1, 200001)
fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(10, 2.9),
                                 gridspec_kw=dict(width_ratios=[1.25, 1, 1]))

# (a) The minimizing sequence: sawtooth functions u_n, tending uniformly to 0.
cols = [ORANGE, GREEN, BLUE, "#7F3FBF"]
for n, c in zip([1, 2, 4, 8], cols):
    ax.plot(x[::50], sawtooth(n, x[::50]), color=c, lw=1.3, label=f"$n={n}$")
ax.axhline(0, color="black", lw=1.6, label=r"limit $u=0$")
ax.set_xlabel("$x$", labelpad=1); ax.set_ylabel("$u_n(x)$")
ax.set_title(r"(a) sawtooth $u_n$, slopes $\pm1$", fontsize=9)
ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper right")
ax.set_ylim(-0.02, 0.62)
ax.spines[["top", "right"]].set_visible(False)

# (b) Energy density as a function of the slope: wells at u' = +-1, hump at 0.
W = np.linspace(-1.6, 1.6, 400)
bx.plot(W, (W**2 - 1)**2, color=BLUE, lw=1.5)
bx.plot([-1, 1], [0, 0], "o", color=ORANGE, ms=5)
bx.plot([0], [1], "o", color=GRAY, ms=5, mfc="white")
bx.annotate("$u'_n$ jumps\nbetween $\\pm1$", xy=(1, 0.02), xytext=(0.15, 1.6),
            fontsize=8, color=ORANGE,
            arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.8))
bx.annotate("limit: $u'=0$", xy=(0, 1), xytext=(-1.55, 2.3), fontsize=8,
            color=GRAY, arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.8))
bx.set_xlabel("slope $u'$", labelpad=1)
bx.set_title(r"(b) non-convex density $((u')^2-1)^2$", fontsize=9)
bx.set_ylim(-0.15, 3)
bx.spines[["top", "right"]].set_visible(False)

# (c) J(u_n) -> 0 = inf J, while the limit u = 0 has J(0) = 1: J is not weakly
# lower semicontinuous.
ns = np.array([1, 2, 4, 8, 16, 32, 64])
Jn = np.array([J_sawtooth(n, x) for n in ns])
cx.loglog(ns, Jn, "o-", color=BLUE, ms=4, label=r"$J(u_n)=1/(12n^2)$")
cx.axhline(1.0, color=ORANGE, lw=1.3, ls="--", label=r"$J(\lim u_n)=J(0)=1$")
cx.set_xlabel("$n$", labelpad=1)
cx.set_title(r"(c) $J(u_n)\to0=\inf J$, never attained", fontsize=9)
cx.legend(frameon=False, fontsize=7, loc="lower left")
cx.set_ylim(1e-5, 4)
cx.grid(True, which="major", alpha=0.3)
cx.spines[["top", "right"]].set_visible(False)

fig.tight_layout(w_pad=1.5)
fig.savefig("../../../figures/ch1/bolza.pdf")
for n, j in zip(ns, Jn):
    print(f"n={n:3d}  J(u_n)={j:.3e}  1/(12n^2)={1/(12*n**2):.3e}")
print("saved figures/ch1/bolza.pdf")
