"""Exercise "Regimes and orders of computation on the truss" (ch6).
Run: cd python/examples/ch6 && python3 exo_truss_regimes.py
"""
import numpy as np
from cy_core import three_bar_truss, cycle, incremental, global_local

# Three-bar truss of Section 6.1.1, Exercise 6.8. (a) The three loadings of
# "Three loadings" (Section 6.1.1) with perfect plasticity: the regime is read on
# the drift of ep per cycle and on its range within a cycle. (b) Two periods of
# the third loading with H = 0.05 E, by the two orders of Section 6.3.
E, sY = 200e3, 200.0                                  # modulus, yield stress (MPa)
eY = sY / E                                           # yield strain, unit of ep
ts = np.linspace(0, 2 * np.pi, 61)                    # one period, 60 steps
# loads Q(t) = (Q1, Q2) in units where N0 = S sY = sY (unit section)
loads = {"(a) Q1 = 1.1 N0 (1 - cos t)": lambda t: np.array([1.1 * sY * (1 - np.cos(t)), 0.0]),
         "(b) Q1 = 1.9 N0 sin t": lambda t: np.array([1.9 * sY * np.sin(t), 0.0]),
         "(c) Q2 = N0, Q1 = 1.2 N0 sin t": lambda t: np.array([1.2 * sY * np.sin(t), sY])}
for name, Q in loads.items():                       # 1. regimes, perfect plasticity
    S = three_bar_truss(E=E, sY=sY, H=0.0, Q=Q)
    z = np.zeros(3)
    for c in range(30):                              # 30 cycles, incremental
        zT, _, _, ep, _ = cycle(S, ts, z)
        drift, z = zT - z, zT                        # ep(T) - ep(0) of the last cycle
    # drift != 0: ratchetting; range != 0 without drift: alternating plasticity;
    # both zero: elastic shakedown (Section 6.5.1, asymptotic regimes)
    rng = ep.max(axis=0) - ep.min(axis=0)
    print(name, "| drift per cycle / eY:", np.round(drift / eY, 3),
          "| range in the cycle / eY:", np.round(rng / eY, 3))

# 2. the two orders on two periods of loading (c), with hardening H = 0.05 E
S = three_bar_truss(E=E, sY=sY, H=0.05 * E, Q=loads["(c) Q2 = N0, Q1 = 1.2 N0 sin t"])
t2 = np.linspace(0, 4 * np.pi, 81)
# "for n, for k": Newton with the consistent tangent (Box 5.3) and the
# initial-strain iteration with the elastic stiffness (Box 6.1)
for method in ["newton", "initial_strain"]:
    it = incremental(S, t2, np.zeros(3), method, tol=1e-9)[3]
    print(f"incremental, {method:14s}: {it.sum():5d} local evaluations, "
          f"{it.min()}..{it.max()} iterations per step")
# "for k, for n": whole history (Box 6.2, initial condition); each iteration
# evaluates the return map at the 80 instants
o = global_local(S, t2, closure="initial", kmax=500, tol=1e-9)
print(f"whole history (Box 6.2)    : {o['iters'] * 80:5d} local evaluations, "
      f"{o['iters']} iterations")
