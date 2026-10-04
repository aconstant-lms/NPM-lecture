"""
Generates figures/ch2/bar_ritz.pdf (Chapter 2, exercise "A bar in tension by
the Ritz method"). Bar clamped at x = 0, uniform load f, free end (F = 0),
EA = L = f = 1: exact solution, one-term (w1 = x) and two-term (w1 = x,
w2 = x^2) Ritz solutions; displacement and normal force N = EA u'.
Figure 2.16, Exercise 2.19. Lengths in units of L, displacements in fL^2/EA,
forces in fL.
Run: cd python/figures/ch2 && python3 fig_bar_ritz.py
"""
import numpy as np
from style_ch2 import plt, BLUE, ORANGE, OUT

# Curves on a fine grid x; the two-term solution is shown as markers on xs.
x = np.linspace(0, 1, 200)
xs = np.linspace(0, 1, 11)
u_ex, N_ex = x - x**2 / 2, 1 - x          # exact: -u'' = 1, u(0) = 0, u'(1) = 0
u_1, N_1 = 0.5 * x, 0.5 + 0 * x           # one term: alpha = (F + fL/2)/EA
u_2, N_2 = xs - xs**2 / 2, 1 - xs         # two terms: the exact solution

fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
# Left: displacements. The one-term solution is exact only at the end x = L.
ax = axes[0]
ax.plot(x, u_ex, color="k", lw=1.4, label="exact")
ax.plot(x, u_1, "--", color=ORANGE, lw=1.4, label=r"Ritz, $w_1=x$")
ax.plot(xs, u_2, "o", color=BLUE, ms=4, mfc="none", label=r"Ritz, $w_1=x,\ w_2=x^2$")
ax.set_xlabel(r"$x/L$")
ax.set_ylabel(r"displacement $u\,EA/(fL^2)$")
ax.set_xlim(0, 1); ax.set_ylim(0, 0.55)
ax.legend(frameon=False, loc="lower right")
ax.set_title("displacement: exact at the end")

# Right: normal forces. One term gives a constant N = 1/2, the average of 1 - x.
ax = axes[1]
ax.fill_between(x, N_ex, N_1, color=ORANGE, alpha=0.12, lw=0)
ax.plot(x, N_ex, color="k", lw=1.4)
ax.plot(x, N_1, "--", color=ORANGE, lw=1.4)
ax.plot(xs, N_2, "o", color=BLUE, ms=4, mfc="none")
ax.annotate("one term: the average\nof the exact force", xy=(0.72, 0.5),
            xytext=(0.5, 0.75), fontsize=8, color=ORANGE,
            arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.8))
ax.set_xlabel(r"$x/L$")
ax.set_ylabel(r"normal force $N/(fL)=EA\,u'/(fL)$")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
ax.set_title("normal force: averaged by one term")
fig.tight_layout()
fig.savefig(OUT + "bar_ritz.pdf")
