"""
Generates figures/ch5/newton.pdf (Chapter 5, Section 5.5).
Thick-walled cylinder (a = 100, b = 200 mm, Voce hardening, p = 320 MPa in 10
steps, 80 elements): residual norm against Newton iteration in load step 6,
with the consistent and with the continuum tangent.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "../../examples/ch5")
from ci_core import Voce
from ci_fem import solve

# Data of Exercise 5.13 (Figure 5.6): E in MPa, Voce law s0 = 250, s_inf = 600 MPa,
# delta = 30
E, nu = 200e3, 0.3
mu, kappa = E / (2 * (1 + nu)), E / (3 * (1 - 2 * nu))
hard = Voce(250.0, 600.0, 30.0)
fig, ax = plt.subplots(figsize=(5.0, 3.4))
# Global Newton (Box 5.3) with the consistent ('alg') or continuum ('cont')
# tangent; residuals of load step 6, divided by the first one
for tg, col, mk, lab in [("alg", "#1F5AC8", "o", "consistent tangent"),
                         ("cont", "#D9822B", "s", "continuum tangent")]:
    *_, its, res = solve(100.0, 200.0, 320.0, 10, 80, mu, kappa, hard, tangent=tg)
    r = np.array(res[5]) / res[5][0]
    ax.semilogy(range(len(r)), r, mk + "-", color=col, ms=5, label=f"{lab} ({sum(its)} iterations in all)")
ax.set_xlabel("Newton iteration")
ax.set_ylabel("relative residual")
ax.grid(alpha=0.25, which="both")
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig("../../../figures/ch5/newton.pdf")
