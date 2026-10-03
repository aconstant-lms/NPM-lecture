"""Exercise "Exactness of the return map" (ch5) of the lecture notes.
Run: cd python/examples/ch5 && python3 exo_exact_1d.py
"""
import numpy as np
from ci_core import return_map_1d
E, sY, K, H = 200e3, 250.0, 2e3, 1e3          # MPa
Eep, eY = E * (K + H) / (E + K + H), sY / E


def path(points, n_sub):                       # n_sub equal steps per segment
    s = ep = al = q = 0.0
    for e0, e1 in zip(points[:-1], points[1:]):
        for eps in np.linspace(e0, e1, n_sub + 1)[1:]:
            s, ep, al, q, _ = return_map_1d(eps, ep, al, q, E, sY, K, H)
    return s, al


exact = sY + Eep * (0.01 - eY)                 # monotone tension, closed form
print(f"monotone to 0.01: exact {exact:.6f}, 1 step {path([0, .01], 1)[0]:.6f},"
      f" 1000 steps {path([0, .01], 1000)[0]:.6f} MPa")
ref = path([0, .004, -.002], 20000)            # 0 -> 0.004 -> -0.002, fine steps
two = path([0, .004, -.002], 1)                # the reversal 0.004 -> -0.002 in ONE step
print(f"reversal inside one step: reference {ref[0]:.6f}, one step {two[0]:.6f} MPa;"
      f" alpha {ref[1]:.6e} / {two[1]:.6e}")
# monotone to 0.01: exact 275.862069, 1 step 275.862069, 1000 steps 275.862069 MPa
# reversal inside one step: reference -262.894028, one step -262.894028 MPa; alpha 6.104249e-03 / 6.104249e-03
