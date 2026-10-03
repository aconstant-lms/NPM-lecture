"""
Generates figures/ch1/bsplines.pdf (Chapter 1, Section "B-splines").
B-spline basis functions computed with the Cox-de Boor recursion on open
uniform knot vectors with 5 knot spans:
  (a) p = 1: the hat functions of linear finite elements;
  (b) p = 2 and (c) p = 3: each function spans p+1 knot spans, the end
      functions are interpolatory, and the functions sum to one;
  (d) a quadratic B-spline curve and its control polygon: the control
      points are not on the curve, except at the ends.
Run: cd python/figures/ch1 && python3 fig_bsplines.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, GREEN, ORANGE, GRAY = "#1F5AC8", "#14963C", "#D85A30", "#666666"


def bspline_basis(knots, p, xi):
    """All N_{i,p}(xi), i = 0..n-1, by the Cox-de Boor recursion (0/0 := 0).
    The last span is closed on the right so that N_{n-1,p}(xi_end) = 1."""
    knots = np.asarray(knots, float)
    m = len(knots) - 1
    xi = np.atleast_1d(xi).astype(float)
    N = np.zeros((m, xi.size))
    for i in range(m):                      # degree 0
        N[i] = (knots[i] <= xi) & (xi < knots[i + 1])
    last = np.max(np.nonzero(knots[:-1] < knots[1:]))
    N[last, xi == knots[-1]] = 1.0
    for q in range(1, p + 1):               # degrees 1..p
        Nq = np.zeros((m - q, xi.size))
        for i in range(m - q):
            d1 = knots[i + q] - knots[i]
            d2 = knots[i + q + 1] - knots[i + 1]
            if d1 > 0:
                Nq[i] += (xi - knots[i]) / d1 * N[i]
            if d2 > 0:
                Nq[i] += (knots[i + q + 1] - xi) / d2 * N[i + 1]
        N = Nq
    return N


def open_uniform(p, nel):
    return np.r_[np.zeros(p), np.arange(nel + 1), np.full(p, nel)]


nel = 5
xi = np.linspace(0, nel, 1001)
fig, axes = plt.subplots(2, 2, figsize=(10, 5.6))
titles = {1: r"(a) $p=1$: hat functions, $C^0$",
          2: r"(b) $p=2$: quadratic B-splines, $C^1$",
          3: r"(c) $p=3$: cubic B-splines, $C^2$"}
for ax, p in zip([axes[0, 0], axes[0, 1], axes[1, 0]], [1, 2, 3]):
    knots = open_uniform(p, nel)
    N = bspline_basis(knots, p, xi)
    for i, Ni in enumerate(N):
        end = i in (0, len(N) - 1)
        ax.plot(xi, Ni, color=ORANGE if end else BLUE, lw=1.6 if end else 1.2)
    k = p + 1 if p > 1 else 2               # an interior function, shaded
    ax.fill_between(xi, N[k], color=GREEN, alpha=0.25, lw=0)
    sup = knots[[k, k + p + 1]]
    ax.annotate("", xy=(sup[0], -0.13), xytext=(sup[1], -0.13),
                arrowprops=dict(arrowstyle="<->", color=GREEN, lw=1))
    ax.text(sup.mean(), -0.24, f"support: {p + 1} knot spans",
            color=GREEN, ha="center", va="top", fontsize=8)
    ax.plot(xi, N.sum(axis=0), "--", color=GRAY, lw=0.9)
    ax.text(nel - 0.6, 1.03, r"$\sum_i N_{i,p}=1$", color=GRAY,
            ha="right", va="bottom", fontsize=8)
    ax.plot(np.unique(knots), np.zeros(nel + 1), "k|", ms=8)
    ax.set_title(titles[p], fontsize=10)
    ax.set_xlim(-0.1, nel + 0.1); ax.set_ylim(-0.42, 1.25)
    ax.set_xlabel(r"$\xi$", labelpad=1)
    ax.set_yticks([0, 0.5, 1])
    ax.text(0.12, 1.06, "interpolatory end function", color=ORANGE,
            fontsize=7, va="bottom", ha="left")
    ax.spines[["top", "right"]].set_visible(False)
    print(f"p={p}: {len(N)} functions, knots {knots.tolist()}, "
          f"max |sum-1| = {np.abs(N.sum(axis=0) - 1).max():.1e}")

# (d) quadratic B-spline curve with its control polygon
ax = axes[1, 1]
p, knots = 2, open_uniform(2, 4)
P = np.array([[0, 0], [0.6, 1.4], [1.8, 1.6], [2.6, 0.2], [3.6, 0.6], [4.2, 1.6]])
t = np.linspace(0, 4, 801)
curve = bspline_basis(knots, p, t).T @ P
ax.plot(P[:, 0], P[:, 1], "o--", color=GRAY, lw=0.9, ms=5, mfc="white")
ax.plot(curve[:, 0], curve[:, 1], color=BLUE, lw=1.8)
tk = np.arange(5)
ck = bspline_basis(knots, p, tk.astype(float)).T @ P
ax.plot(ck[:, 0], ck[:, 1], "s", color=BLUE, ms=3.5)
for i, (x, y) in enumerate(P):
    ax.text(x + 0.08, y + 0.08, rf"$\boldsymbol{{P}}_{{{i + 1}}}$", fontsize=8)
ax.set_title("(d) quadratic B-spline curve and control polygon", fontsize=10)
ax.set_aspect("equal"); ax.set_xlim(-0.3, 4.7); ax.set_ylim(-0.3, 2.0)
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)
ax.text(2.1, -0.2, "squares: images of the knots (element ends)",
        fontsize=7, color=BLUE, ha="center")

fig.tight_layout(h_pad=1.2)
fig.savefig("../../../figures/ch1/bsplines.pdf")
print("saved figures/ch1/bsplines.pdf")
