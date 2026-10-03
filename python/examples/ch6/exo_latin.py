"""Exercise "LATIN on the truss: directions and relaxation" (ch6).
Run: cd python/examples/ch6 && python3 exo_latin.py
"""
import numpy as np
from cy_core import three_bar_truss, incremental, latin

E, sY = 200e3, 200.0
S = three_bar_truss(E=E, sY=sY, H=0.05 * E,
                    Q=lambda t: np.array([1.2 * sY * np.sin(t), sY]))
ts = np.linspace(0, 4 * np.pi, 81)                    # window of two periods
ref = incremental(S, ts, np.zeros(3), tol=1e-12)[2]   # step-by-step reference
for hE in [1.0, 4.0]:
    for mu in [0.0, 0.3]:
        o = latin(S, ts, hE / E, mu=mu, kmax=400, tol=1e-10)
        err = np.max(np.abs(o["ep"] - ref)) / (sY / E)
        print(f"hE = {hE:3.0f}, mu = {mu:.1f}: {o['iters']:3d} iterations, "
              f"indicator {o['err'][-1]:.1e}, local iterate error {err:.1e} eY")
o = latin(S, ts, 1 / E, mu=0.0, kmax=400, tol=0.0)
print("mu = 0, last indicators:", np.round(o["err"][-4:], 5))
