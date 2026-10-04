"""Exercise "Contractivity of the return map" (ch5) of the lecture notes.
Run: cd python/examples/ch5 && python3 exo_contract.py
"""
# The return map never increases distances in the energy norm (Section 5.9);
# forward Euler does not have this property. Data of Figure 5.12.
import numpy as np
from ci_core import radial_return, Linear, dev, norm, R23
E, nu, sY = 200e3, 0.3, 250.0                  # MPa, -, MPa
mu, kappa = E / (2 * (1 + nu)), E / (3 * (1 - 2 * nu))
hard = Linear(sY, 0.0)                          # perfect plasticity


def energy_dist(s1, s2):                        # ||s1 - s2|| in the norm of S = C^-1
    """Energy distance sqrt(d : S : d), d = s1 - s2, in sqrt(MPa):
    deviatoric part ||dev d||^2 / (2 mu) plus pressure part p^2 / kappa."""
    d = s1 - s2
    return np.sqrt(norm(dev(d))**2 / (2 * mu) + (d[:3].sum() / 3)**2 / kappa)


def forward(eps1, eps0, ep):                    # forward Euler, continuum tangent
    """Explicit step eps0 -> eps1 from the plastic strain ep: the normal and the
    loading test are taken at the start of the step; no return to the surface."""
    s0 = 2 * mu * dev(eps0 - ep)
    n = s0 / norm(s0) if norm(s0) > 0 else 0 * s0
    de = eps1 - eps0
    on = norm(s0) >= R23 * sY * (1 - 1e-9)      # on the yield surface at t_n
    # perfect plasticity: Delta gamma = <n : Delta eps> (shear terms counted twice)
    dg = max(np.sum(n * de * [1, 1, 1, 2, 2, 2]), 0.0) if on else 0.0
    ep = ep + dg * n
    sig = kappa * eps1[:3].sum() * np.r_[1, 1, 1, 0, 0, 0] + 2 * mu * dev(eps1 - ep)
    return sig, ep


# (c) state B: plastic tension e11 = 0.004 (lateral -0.002), then back to zero strain
epB = np.zeros(6)
for t in np.linspace(0, 1, 200)[1:]:
    _, epB, *_ = radial_return(t * np.array([.004, -.002, -.002, 0, 0, 0]), epB, 0.0,
                               np.zeros(6), mu, kappa, hard)
_, epB, *_ = radial_return(np.zeros(6), epB, 0.0, np.zeros(6), mu, kappa, hard)
gam = np.array([0, 0, 0, 0.5, 0, 0])            # e12 = gamma / 2
# shear A (virgin) and B to gamma = 0.03 in nstep steps: zA, zB plastic strains with
# backward Euler (radial return), yA, yB with forward Euler
for nstep in (4, 20, 500):
    drift = 0.0
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
        # overshoot of the yield surface by forward Euler:
        # ||dev sigma|| / (sqrt(2/3) sY) - 1
        drift = max(drift, norm(dev(tA)) / (R23 * sY) - 1) if i > 0 else 0.0
    mono = all(np.diff(dBE) <= 1e-12)           # distance non-increasing at every step
    print(f"{nstep:3d} steps: backward Euler {dBE[0]:.4e} -> {dBE[-1]:.4e} "
          f"(non-increasing: {mono}); forward Euler -> {dFE[-1]:.4e},"
          f" largest overshoot of the yield surface {100 * drift:.1f} %")
# (b) filament, perfect plasticity: A on the surface, B inside by 10 MPa, de = 2.5e-4
# forward Euler moves B elastically by E de = 50 MPa, past the yield stress
sA, sB, d = sY, sY - 10.0, E * 2.5e-4
print(f"filament: distance {sB - sA:+.1f} -> forward Euler {sB + d - sA:+.1f},"
      f" backward Euler {min(sB + d, sY) - sA:+.1f} MPa")
#   4 steps: backward Euler 5.2042e-01 -> 1.0173e-03 (non-increasing: True); forward Euler -> 1.2294e-01, largest overshoot of the yield surface 299.7 %
#  20 steps: backward Euler 5.2042e-01 -> 6.6965e-06 (non-increasing: True); forward Euler -> 6.3589e-02, largest overshoot of the yield surface 59.9 %
# 500 steps: backward Euler 5.2042e-01 -> 1.5058e-07 (non-increasing: True); forward Euler -> 3.7087e-03, largest overshoot of the yield surface 2.3 %
# filament: distance -10.0 -> forward Euler +40.0, backward Euler +0.0 MPa
