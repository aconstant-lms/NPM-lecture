"""Exercise "Cyclic loading and the Bauschinger effect" (ch4) of the lecture notes.
Run: python3 python/examples/ch4/exo_cyclic_1d.py
"""
import numpy as np
E, sY = 200e3, 250.0                    # MPa
eY = sY / E

def cycle(K, H, emax=3 * eY, n=2000):
    ep = q = al = 0.0
    out = []
    for a, b in [(0, emax), (emax, -emax), (-emax, emax)]:
        for eps in np.linspace(a, b, n)[1:]:
            s_tr = E * (eps - ep)                # elastic predictor
            f_tr = abs(s_tr - q) - (sY + K * al)
            if f_tr > 0:                         # plastic corrector (exact here)
                dg = f_tr / (E + K + H)
                sg = np.sign(s_tr - q)
                ep, q, al = ep + dg * sg, q + H * dg * sg, al + dg
        out.append(E * (eps - ep))               # stress at the end of each branch
    return np.round(out, 3)

for K, H in [(3e3, 0.0), (0.0, 3e3), (1.5e3, 1.5e3)]:
    print(f"K={K:6.0f} H={H:6.0f}  peaks:", cycle(K, H))
# K=  3000 H=     0  peaks: [ 257.389 -271.949  286.079]
# K=     0 H=  3000  peaks: [ 257.389 -257.389  257.389]
# K=  1500 H=  1500  peaks: [ 257.389 -264.669  271.842]
