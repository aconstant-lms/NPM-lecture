"""Exercise "LATIN on the truss: directions and relaxation" (ch6).
Run: cd python/examples/ch6 && python3 exo_latin.py
"""
import numpy as np
from cy_core import three_bar_truss, incremental, latin

# Exercise 6.9: Box 6.3 on the truss, third loading of Section 6.1.1 with
# H = 0.05 E, window of two periods with the initial condition (Figure 6.9).
E, sY = 200e3, 200.0                                  # modulus, yield stress (MPa)
S = three_bar_truss(E=E, sY=sY, H=0.05 * E,
                    Q=lambda t: np.array([1.2 * sY * np.sin(t), sY]))
ts = np.linspace(0, 4 * np.pi, 81)                    # window of two periods
ref = incremental(S, ts, np.zeros(3), tol=1e-12)[2]   # step-by-step reference
# search direction h (times E: hE = 1 or 4) and relaxation mu (Box 6.3, step 4)
for hE in [1.0, 4.0]:
    for mu in [0.0, 0.3]:
        o = latin(S, ts, hE / E, mu=mu, kmax=400, tol=1e-10)
        # error of the constitutive iterate ep_hat against the reference, in eY
        err = np.max(np.abs(o["ep"] - ref)) / (sY / E)
        print(f"hE = {hE:3.0f}, mu = {mu:.1f}: {o['iters']:3d} iterations, "
              f"indicator {o['err'][-1]:.1e}, local iterate error {err:.1e} eY")
# without relaxation the indicator stalls at a constant value: only averages
# converge in statics (box "Convergence of the LATIN method", Section 6.4.2)
o = latin(S, ts, 1 / E, mu=0.0, kmax=400, tol=0.0)
print("mu = 0, last indicators:", np.round(o["err"][-4:], 5))
