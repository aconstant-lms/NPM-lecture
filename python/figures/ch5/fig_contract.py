"""
Generates figures/ch5/contract.pdf (Chapter 5, exercise "Contractivity of the
return map"). Perfect J2 plasticity; state A virgin, state B after plastic
tension and return to zero strain; both are sheared to gamma = 0.03 in n steps.
Energy distance between the two stresses against the step number, backward
Euler (return map) and forward Euler. Same data as exo_contract.py.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "../../examples/ch5")
from ci_core import radial_return, Linear, dev, norm, R23

blue, orange = "#1F5AC8", "#D9822B"
E, nu, sY = 200e3, 0.3, 250.0
mu, kappa = E / (2 * (1 + nu)), E / (3 * (1 - 2 * nu))
hard = Linear(sY, 0.0)


def energy_dist(s1, s2):
    d = s1 - s2
    return np.sqrt(norm(dev(d))**2 / (2 * mu) + (d[:3].sum() / 3)**2 / kappa)


def forward(eps1, eps0, ep):
    s0 = 2 * mu * dev(eps0 - ep)
    n = s0 / norm(s0) if norm(s0) > 0 else 0 * s0
    de = eps1 - eps0
    on = norm(s0) >= R23 * sY * (1 - 1e-9)
    dg = max(np.sum(n * de * [1, 1, 1, 2, 2, 2]), 0.0) if on else 0.0
    ep = ep + dg * n
    sig = kappa * eps1[:3].sum() * np.r_[1, 1, 1, 0, 0, 0] + 2 * mu * dev(eps1 - ep)
    return sig, ep


epB = np.zeros(6)
for t in np.linspace(0, 1, 200)[1:]:
    _, epB, *_ = radial_return(t * np.array([.004, -.002, -.002, 0, 0, 0]), epB, 0.0,
                               np.zeros(6), mu, kappa, hard)
_, epB, *_ = radial_return(np.zeros(6), epB, 0.0, np.zeros(6), mu, kappa, hard)
gam = np.array([0, 0, 0, 0.5, 0, 0])

fig, ax = plt.subplots(figsize=(5.4, 3.5))
for nstep, mk, ls in [(4, "o", "-"), (20, "s", "-"), (500, "", "-")]:
    zA, zB, yA, yB = np.zeros(6), epB.copy(), np.zeros(6), epB.copy()
    dBE, dFE = [], []
    for i in range(nstep + 1):
        eps = 0.03 * i / nstep * gam
        sA, zA, *_ = radial_return(eps, zA, 0.0, np.zeros(6), mu, kappa, hard)
        sB, zB, *_ = radial_return(eps, zB, 0.0, np.zeros(6), mu, kappa, hard)
        if i > 0:
            tA, yA = forward(eps, eps - 0.03 / nstep * gam, yA)
            tB, yB = forward(eps, eps - 0.03 / nstep * gam, yB)
        else:
            tA, tB = sA, sB
        dBE.append(energy_dist(sA, sB)); dFE.append(energy_dist(tA, tB))
    i = np.arange(1, nstep + 1)
    ax.loglog(i, dBE[1:], mk + ls, color=blue, ms=4, lw=1.3)
    ax.loglog(i, dFE[1:], mk + "--", color=orange, ms=4, lw=1.1, mfc="white")
    ax.annotate(f"$n={nstep}$", (i[-1], dBE[-1]), xytext=(4, 0),
                textcoords="offset points", fontsize=8, color=blue, va="center")
    ax.annotate(f"$n={nstep}$", (i[-1], dFE[-1]), xytext=(4, 7),
                textcoords="offset points", fontsize=8, color=orange, va="center")
ax.axhline(dBE[0], color="0.5", lw=0.8, ls=":")
ax.text(1.05, dBE[0] * 1.5, "initial distance", fontsize=8, color="0.4")
ax.plot([], [], "-", color=blue, label="backward Euler (return map)")
ax.plot([], [], "--", color=orange, label="forward Euler")
ax.set_xlim(0.8, 2000)
ax.set_ylim(None, 3.0)
ax.set_xlabel("step number $i$")
ax.set_ylabel(r"$\|\sigma_A-\sigma_B\|_E$  ($\sqrt{\mathrm{MPa}}$)")
ax.grid(alpha=0.25, which="both")
ax.legend(fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig("../../../figures/ch5/contract.pdf")
