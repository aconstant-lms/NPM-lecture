"""Exercise "Nonlinear hardening: Armstrong-Frederick and Voce" (ch4) of the lecture notes.
Run: python3 python/examples/ch4/exo_af_voce.py
"""
# Two checks of the nonlinear hardening laws of Section 4.3.5 in monotone tension:
# the Armstrong-Frederick back stress, and the inverse of the Voce law.
import numpy as np
from scipy.integrate import solve_ivp
H, zeta, sY = 20e3, 100.0, 250.0         # MPa, -, MPa
# H: kinematic modulus, zeta: recall coefficient, sY: initial yield stress
# (a) Armstrong-Frederick, q_dot = H ep_dot - zeta q |ep_dot|
#     (in tension |ep_dot| = ep_dot)
# monotone tension: dq/dep = H - zeta q, closed form q = H/zeta (1 - exp(-zeta ep))
sol = solve_ivp(lambda ep, q: H - zeta * q, [0, 0.05], [0.0], rtol=1e-12, atol=1e-12)
# q at eps^p = 0.05: numeric, closed form, and the saturation value H/zeta
print("q(0.05) numeric %.6f  closed form %.6f  saturation %.1f"
      % (sol.y[0, -1], H / zeta * (1 - np.exp(-zeta * 0.05)), H / zeta))
# (c) Voce: R(alpha) = sinf - (sinf - s0) exp(-d alpha); in tension alpha = eps^p
# and sigma = R(alpha), so eps = sigma/E - ln((sinf - sigma)/(sinf - s0)) / d.
# Voce: invert sigma(eps) in tension and check against the forward law
E, s0, sinf, d = 200e3, 250.0, 600.0, 20.0   # MPa, MPa, MPa (saturation), - (rate)
sig = np.array([300.0, 400.0, 500.0])        # test stresses (MPa)
eps = sig / E - np.log((sinf - sig) / (sinf - s0)) / d
alpha = eps - sig / E                        # alpha = eps^p = eps - sigma/E
# R(alpha) - sigma must vanish
print("Voce check:", np.round(sinf - (sinf - s0) * np.exp(-d * alpha) - sig, 12))
# q(0.05) numeric 198.652411  closed form 198.652411  saturation 200.0
# Voce check: [0. 0. 0.]
