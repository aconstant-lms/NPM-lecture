"""Exercise "The Bree vessel: period map and direct cyclic method" (ch6).
Run: cd python/examples/ch6 && python3 exo_bree_cyclic.py
"""
import numpy as np
from cy_core import bree_vessel, cycle, period_map, fixed_point, dcm_sweep_map, melan

s0, Ep = 280.0, 200e3 / 0.7                           # sigma_0, plane modulus E/(1-nu)
eY = s0 / Ep
lam = lambda t: abs(np.sin(t))                        # heat flux switched on and off
ts = np.linspace(0, np.pi, 41)                        # one thermal cycle, 40 steps


def vessel(X, Y, Hr):
    return bree_vessel(X * s0, Y * s0, lam, E=Ep, sY=s0, H=Hr * Ep, nlay=100)


# 1. ratchetting, H = 0: drift of the mean, constant residual stress
S = vessel(0.7, 1.5, 0.0)
_, Z = melan(S)
z = np.zeros(S.nf)
for c in range(20):
    zT = cycle(S, ts, z)[0]
    dmean, drho, z = np.mean(zT - z), np.max(np.abs(Z @ (zT - z))), zT
print(f"X=0.7, Y=1.5, H=0: drift of <ep> per cycle {dmean / eY:.4f} eY, "
      f"change of residual stress {drho / s0:.1e} s0")
# 2. slow shakedown, H = 0.02 E: three iterations on the period map, and DCM
S = vessel(0.7, 1.5, 0.02)
Pi, z0 = period_map(S, ts), np.zeros(S.nf)
for scheme in ["picard", "km", "anderson"]:
    Zs, r = fixed_point(Pi, z0, scheme, theta=0.5, m=5, kmax=400, tol=1e-10 * eY)
    print(f"period map, {scheme:8s}: {len(r):3d} cycles, residual {r[-1] / eY:.1e} eY")
    if scheme == "picard":
        zP = Zs[-1]                                    # cycle-by-cycle limit
M = dcm_sweep_map(S, ts)
for scheme in ["picard", "anderson"]:
    H_, r = fixed_point(M, np.zeros(len(ts) * S.nf), scheme, m=5, kmax=400,
                        tol=1e-10 * eY)
    zD = H_[-1].reshape(len(ts), S.nf)[0]
    print(f"direct cyclic, {scheme:8s}: {len(r):3d} sweeps, residual {r[-1] / eY:.1e} eY")
d = zD - zP                                            # two periodic states
print(f"DCM+Anderson vs cycles: |Pi(zD)-zD| = {np.max(np.abs(Pi(zD) - zD)) / eY:.1e} eY, "
      f"mean difference {d.mean() / eY:.3f} eY, "
      f"residual-stress difference {np.max(np.abs(Z @ d)) / s0:.4f} s0")
