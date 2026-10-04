"""
Generates figures/ch5/isoerror.pdf (Chapter 5, Section 5.9).
Iso-error map of the radial return, perfect J2 plasticity: start on the yield
surface at uniaxial tension, apply in ONE step the strain increment
de = (a n + b t) R/(2 mu), n normal, t a unit deviatoric tangent, R the radius
of the von Mises circle. Error = relative error of the deviatoric stress
against 1000 substeps of the same increment.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "../../examples/ch5")
from ci_core import radial_return, Linear, dev, norm, R23

# Material (MPa), perfect plasticity; R = sqrt(2/3) sY, radius of the von Mises circle
E, nu, sY = 200e3, 0.3, 250.0
mu, kappa = E / (2 * (1 + nu)), E / (3 * (1 - 2 * nu))
hard, R = Linear(sY, 0.0), R23 * sY
n = np.array([2, -1, -1, 0, 0, 0]) / np.sqrt(6)            # uniaxial direction
t = np.array([0, 1, -1, 0, 0, 0]) / np.sqrt(2)             # orthogonal deviator
eps0 = R / (2 * mu) * n                                    # on the yield surface
ep0 = np.zeros(6)


# Deviatoric stress after the increment de, applied in nsub equal sub-steps of
# the radial return (Box 5.2)
def stress(de, nsub):
    ep = ep0
    for k in range(1, nsub + 1):
        s, ep, *_ = radial_return(eps0 + k / nsub * de, ep, 0.0, np.zeros(6), mu, kappa, hard)
    return dev(s)


# Grid of normal (a) and tangential (b) increments, in units of the yield strain
# R/(2 mu); error in % against 1000 sub-steps
A = np.linspace(0, 6, 41)
B = np.linspace(0, 6, 41)
err = np.zeros((len(B), len(A)))
for i, b in enumerate(B):
    for j, a in enumerate(A):
        de = (a * n + b * t) * R / (2 * mu)
        ex = stress(de, 1000)
        err[i, j] = 100 * norm(stress(de, 1) - ex) / norm(ex)
fig, ax = plt.subplots(figsize=(4.6, 4.0))
cs = ax.contour(A, B, err, levels=[2, 5, 10, 15, 20], colors="#1F5AC8", linewidths=1.2)
ax.clabel(cs, fmt="%d %%", fontsize=8)
ax.set_xlabel(r"normal increment $a$")
ax.set_ylabel(r"tangential increment $b$")
ax.set_aspect("equal")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("../../../figures/ch5/isoerror.pdf")
print("max error %.2f %% at a = b = 6; error at (a,b)=(0,1): %.2f %%, (0,3): %.2f %%"
      % (err[-1, -1], err[np.argmin(abs(B-1)), 0], err[np.argmin(abs(B-3)), 0]))
# max error 7.66 % at a = b = 6; error at (a,b)=(0,1): 8.76 %, (0,3): 22.15 %
