"""
Generates figures/ch2/conditioning.pdf (Chapter 2, exercise "Why not
polynomials?"). Condition number of the stiffness matrix of the bar
(EA = L = 1, clamped at 0, free at 1) against the number n of trial
functions: monomials x^m (computed in 60-digit arithmetic) and n hat
functions on a uniform mesh.
Figure 2.17, Exercise 2.20 (the same matrices as exo_conditioning.py, for
n = 1 .. 15).
Run: cd python/figures/ch2 && python3 fig_conditioning.py
"""
import numpy as np
import mpmath as mp
from style_ch2 import plt, BLUE, ORANGE, GRAY, OUT

# 60 digits: double precision cannot resolve the smallest eigenvalue of the
# monomial matrix beyond n ~ 12 (its condition number exceeds 1/eps).
mp.mp.dps = 60
ns = np.arange(1, 16)
cm, ch = [], []
for n in ns:
    # Monomials w_m = x^m: K_ml = int_0^1 w_m' w_l' dx = m l / (m + l - 1).
    K = mp.matrix(n, n)
    for m in range(1, n + 1):
        for l in range(1, n + 1):
            K[m - 1, l - 1] = mp.mpf(m * l) / (m + l - 1)
    ev = mp.eigsy(K, eigvals_only=True)
    cm.append(float(max(ev) / min(ev)))
    # n hat functions, h = 1/n: (1/h) tridiag(-1, 2, -1) with one element at the free end.
    H = n * (2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1))
    H[-1, -1] = n                                    # free end
    e = np.linalg.eigvalsh(H)
    ch.append(e[-1] / e[0])

fig, ax = plt.subplots(figsize=(4.6, 3.0))
ax.semilogy(ns, cm, "o-", color=ORANGE, ms=4, label=r"monomials $x^m$")
ax.semilogy(ns, ch, "s-", color=BLUE, ms=4, label="hat functions")
# Dashed: 1 / (double-precision machine epsilon) ~ 4.5e15.
ax.axhline(1 / np.finfo(float).eps, color=GRAY, lw=0.8, ls="--")
ax.text(1.2, 1.8 / np.finfo(float).eps, "1 / machine precision", fontsize=7, color=GRAY)
ax.set_xlabel(r"number of trial functions $n$")
ax.set_ylabel(r"condition number of $\mathbf{K}$")
ax.set_xticks(ns[::2])
ax.set_ylim(1, 1e22)
ax.grid(alpha=0.25, which="major")
ax.legend(frameon=False, loc="center right")
fig.tight_layout()
fig.savefig(OUT + "conditioning.pdf")
for n, a, b in zip(ns, cm, ch):
    print(n, f"{a:.2e}", f"{b:.1f}")
