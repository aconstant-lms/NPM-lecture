"""
Generates figures/ch5/errors.pdf (Chapter 5, exercises "A non-monotone step"
and "Simple shear, then a non-proportional path").
Relative errors of the return map against the number of steps n, log-log:
(a) filament, strain path with a turning point inside a step;
(b) J2 radial return, shear then tension at fixed shear (second stage in n steps).
Same data and reference solutions as exo_nonmonotone_1d.py and exo_shear.py.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "../../examples/ch5")
from ci_core import return_map_1d, radial_return, Linear

blue, orange, green = "#1F5AC8", "#D9822B", "#14963C"

# (a) filament, data of exo_nonmonotone_1d.py
E, sY, K, H = 200e3, 250.0, 2e3, 1e3
tp = 1 / np.sqrt(2)
eps = lambda t: np.where(t <= tp, 0.008 * t / tp, 0.008 - 0.010 * (t - tp) / (2 - tp))


# n uniform steps of the return map (Box 5.1) on [0, 2]
def run1d(n):
    s = ep = al = q = 0.0
    for t in np.linspace(0, 2, n + 1)[1:]:
        s, ep, al, q, _ = return_map_1d(eps(t), ep, al, q, E, sY, K, H)
    return s, al


# Reference with 400001 steps, then relative errors for about 30 values of n
sr, ar = run1d(400001)
n1 = np.unique(np.round(np.logspace(0, np.log10(2048), 34)).astype(int))
e1 = np.array([[abs(run1d(n)[0] - sr) / abs(sr), abs(run1d(n)[1] - ar) / ar] for n in n1])

# (b) shear then tension, data of exo_shear.py
nu = 0.3
mu, kappa = E / (2 * (1 + nu)), E / (3 * (1 - 2 * nu))
hard = Linear(sY, 2e3)
# Stage 1: shear to e12 = 0.002 in 500 steps (fixed); stage 2: e11 0 -> 0.004
A = np.array([0, 0, 0, 0.002, 0, 0])
B = A + np.array([0.004, 0, 0, 0, 0, 0])
ep0, al0, be0 = np.zeros(6), 0.0, np.zeros(6)
for t in np.linspace(0, 1, 501)[1:]:
    _, ep0, al0, be0, *_ = radial_return(t * A, ep0, al0, be0, mu, kappa, hard)


# Second stage in n steps of the radial return (Box 5.2), from the state after
# stage 1
def two_stage(n):
    ep, al, beta = ep0, al0, be0
    for t in np.linspace(0, 1, n + 1)[1:]:
        s, ep, al, beta, *_ = radial_return(A + t * (B - A), ep, al, beta, mu, kappa, hard)
    return s, al


# Reference with 200000 steps; relative errors on sigma_11, sigma_12, alpha
s2, a2 = two_stage(200000)
n2 = np.unique(np.round(np.logspace(0, 3, 22)).astype(int))
e2 = []
for n in n2:
    s, a = two_stage(n)
    e2.append([abs(s[0] - s2[0]) / abs(s2[0]), abs(s[3] - s2[3]) / abs(s2[3]), abs(a - a2) / a2])
e2 = np.array(e2)

fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3))
ax = axes[0]
ax.loglog(n1, e1[:, 0], "o-", color=blue, ms=3.5, lw=1.2, label=r"stress $\sigma$")
ax.loglog(n1, e1[:, 1], "s-", color=orange, ms=3.5, lw=1.2, label=r"$\alpha$")
ax.set_title("(a) filament, turning point inside a step", fontsize=10)
ax = axes[1]
ax.loglog(n2, e2[:, 1], "o-", color=blue, ms=3.5, lw=1.2, label=r"$\sigma_{12}$")
ax.loglog(n2, e2[:, 0], "^-", color=green, ms=3.5, lw=1.2, label=r"$\sigma_{11}$")
ax.loglog(n2, e2[:, 2], "s-", color=orange, ms=3.5, lw=1.2, label=r"$\alpha$")
ax.set_title("(b) $J_2$, shear then tension", fontsize=10)
# Dashed reference lines of slope -1 (first order)
for ax, nn, c0 in [(axes[0], n1, 0.5), (axes[1], n2, 0.6)]:
    x = np.array([nn[0], nn[-1]], float)
    ax.loglog(x, c0 / x, "--", color="0.5", lw=1.0, label=r"slope $-1$")
    ax.set_xlabel("number of steps $n$")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8, loc="lower left")
axes[0].set_ylabel("relative error")
fig.tight_layout()
fig.savefig("../../../figures/ch5/errors.pdf")
print(f"(a) reference sigma = {sr:.5f}, alpha = {ar:.6e}; (b) s12 = {s2[3]:.4f}")
