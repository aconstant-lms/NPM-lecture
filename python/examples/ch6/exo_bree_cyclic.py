"""Exercise "The Bree vessel: period map and direct cyclic method" (ch6).
Run: cd python/examples/ch6 && python3 exo_bree_cyclic.py
"""
import numpy as np
from cy_core import bree_vessel, cycle, period_map, fixed_point, dcm_sweep_map, melan

# Exercise 6.10: Bree case (a), X = sigma_P/sY = 0.7, Y = sigma_T/sY = 1.5
# (Figure 6.4), 100 layers, 40 steps per cycle. Sections 6.5 and 6.6.
sY, Ep = 280.0, 200e3 / 0.7                           # sY, plane modulus E/(1-nu)
eY = sY / Ep                                          # yield strain, unit of ep
lam = lambda t: abs(np.sin(t))                        # heat flux switched on and off
ts = np.linspace(0, np.pi, 41)                        # one thermal cycle, 40 steps


def vessel(X, Y, Hr):
    """Bree vessel with sigma_P = X sY, sigma_T = Y sY and H = Hr E (MPa)."""
    return bree_vessel(X * sY, Y * sY, lam, E=Ep, sY=sY, H=Hr * Ep, nlay=100)


# 1. ratchetting, H = 0: drift of the mean, constant residual stress
# The drift is the uniform part <ep>, in the kernel of Z: rho = -Z ep does not
# see it (Melan decomposition, (6.10)).
S = vessel(0.7, 1.5, 0.0)
_, Z = melan(S)
z = np.zeros(S.nf)
for c in range(20):
    zT = cycle(S, ts, z)[0]
    dmean, drho, z = np.mean(zT - z), np.max(np.abs(Z @ (zT - z))), zT
print(f"X=0.7, Y=1.5, H=0: drift of <ep> per cycle {dmean / eY:.4f} eY, "
      f"change of residual stress {drho / sY:.1e} sY")
# 2. slow shakedown, H = 0.02 E: three iterations on the period map, and DCM
# Period map Pi: ep(0) -> ep(T) (Section 6.5.2), iterated by Picard (cycle by
# cycle), Krasnoselskii-Mann (theta = 1/2) and Anderson with depth 5 (Appendix E).
S = vessel(0.7, 1.5, 0.02)
Pi, z0 = period_map(S, ts), np.zeros(S.nf)
for scheme in ["picard", "km", "anderson"]:
    Zs, r = fixed_point(Pi, z0, scheme, theta=0.5, m=5, kmax=400, tol=1e-10 * eY)
    print(f"period map, {scheme:8s}: {len(r):3d} cycles, residual {r[-1] / eY:.1e} eY")
    if scheme == "picard":
        zP = Zs[-1]                                    # cycle-by-cycle limit
# Direct cyclic method (Box 6.2, periodic closure) as a map on the whole history,
# plain (Picard on sweeps) and accelerated by Anderson (Algorithm 6.1)
M = dcm_sweep_map(S, ts)
for scheme in ["picard", "anderson"]:
    H_, r = fixed_point(M, np.zeros(len(ts) * S.nf), scheme, m=5, kmax=400,
                        tol=1e-10 * eY)
    zD = H_[-1].reshape(len(ts), S.nf)[0]              # ep(t_0) of the last iterate
    print(f"direct cyclic, {scheme:8s}: {len(r):3d} sweeps, residual {r[-1] / eY:.1e} eY")
# 3. two periodic states: both are fixed points of Pi; they differ by a nearly
# uniform plastic strain (kernel of Z) and slightly in the residual stress
d = zD - zP                                            # two periodic states
print(f"DCM+Anderson vs cycles: |Pi(zD)-zD| = {np.max(np.abs(Pi(zD) - zD)) / eY:.1e} eY, "
      f"mean difference {d.mean() / eY:.3f} eY, "
      f"residual-stress difference {np.max(np.abs(Z @ d)) / sY:.4f} sY")
