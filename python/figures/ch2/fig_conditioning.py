"""
Generates figures/ch2/conditioning.pdf (Chapter 2, exercise "Why not
polynomials?"). Condition number of the stiffness matrix of the bar
(EA = L = 1, clamped at 0, free at 1) against the number n of trial
functions: monomials x^m (computed in 60-digit arithmetic) and n hat
functions on a uniform mesh.
"""
import numpy as np
import mpmath as mp
from style_ch2 import plt, BLUE, ORANGE, GRAY, OUT

mp.mp.dps = 60
ns = np.arange(1, 16)
cm, ch = [], []
for n in ns:
    K = mp.matrix(n, n)
    for m in range(1, n + 1):
        for l in range(1, n + 1):
            K[m - 1, l - 1] = mp.mpf(m * l) / (m + l - 1)
    ev = mp.eigsy(K, eigvals_only=True)
    cm.append(float(max(ev) / min(ev)))
    H = n * (2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1))
    H[-1, -1] = n                                    # free end
    e = np.linalg.eigvalsh(H)
    ch.append(e[-1] / e[0])

fig, ax = plt.subplots(figsize=(4.6, 3.0))
ax.semilogy(ns, cm, "o-", color=ORANGE, ms=4, label=r"monomials $x^m$")
ax.semilogy(ns, ch, "s-", color=BLUE, ms=4, label="hat functions")
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
