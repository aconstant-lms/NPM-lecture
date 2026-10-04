"""Exercise "Zarka's estimate of the shakedown state" (ch6).
Run: cd python/examples/ch6 && python3 exo_zarka.py
"""
import numpy as np
from cy_core import bree_vessel, incremental, period_map, fixed_point, melan, zarka

# Exercise 6.11: Box 6.5 on the Bree vessel with H = 0.02 E, compared with the
# limit of the cycle-by-cycle computation (Section 6.7, Figure 6.14).
s0, Ep = 280.0, 200e3 / 0.7                           # sY (MPa), plane modulus E/(1-nu)
eY = s0 / Ep
ts = np.linspace(0, np.pi, 41)                        # one thermal cycle, 40 steps
for X, Y in [(0.6, 1.5), (0.7, 1.5)]:
    S = bree_vessel(X * s0, Y * s0, lambda t: abs(np.sin(t)), E=Ep, sY=s0,
                    H=0.02 * Ep, nlay=100)
    _, Z = melan(S)
    z0 = np.zeros(S.nf)
    # Box 6.5, step 1: elastic stress over the cycle (zero plastic strain)
    sel = np.array([S.solve_elastic(t, z0)[2] for t in ts])   # elastic history
    # step 3: first half-cycle (t from 0 to T/2, hottest instant) computed
    # incrementally, then Y1 = (H I + Z) ep1, the transformed parameter (6.17)
    ep1 = incremental(S, ts[:21], z0)[2][-1]                 # first half-cycle
    Y1 = S.H * ep1 + Z @ ep1                                 # transformed parameter
    # steps 2, 4, 5: admissible interval, projection, linear problem for ep
    ep_z, rho, Yz, ok = zarka(S, sel, Y1)
    # reference: cycle by cycle (Picard on the period map) to 1e-10 eY
    Pi = period_map(S, ts)
    Zs, r = fixed_point(Pi, z0, "picard", kmax=400, tol=1e-10 * eY)
    print(f"X={X}, Y={Y}: elastic shakedown possible at every layer: {ok.all()}; "
          f"cycles to converge {len(r)}; "
          f"max|ep_Zarka - ep_cycles| = {np.max(np.abs(ep_z - Zs[-1])) / eY:.1e} eY; "
          f"|Pi(ep_Zarka) - ep_Zarka| = {np.max(np.abs(Pi(ep_z) - ep_z)) / eY:.1e} eY")
