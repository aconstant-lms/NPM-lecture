"""Exercise "The viscoplastic filament" (ch4) of the lecture notes.
Run: python3 python/examples/ch4/exo_perzyna.py
"""
# Perzyna filament without hardening (Section 4.2.5): relaxation at fixed strain,
# and the steady overstress eta c under a constant strain rate c.
import numpy as np
from scipy.integrate import solve_ivp
E, sY, eta = 200e3, 250.0, 1000.0       # MPa, MPa, MPa.s
tau = eta / E                             # relaxation time [s]

def rhs(t, s, rate):                      # d sigma/dt = E (eps_dot - <|s|-sY>/eta sign s)
    """Right-hand side of the Perzyna law (4.17) for sigma = s[0] (MPa);
    rate is the imposed strain rate (1/s). <x> is the positive part."""
    return [E * rate - E / eta * max(abs(s[0]) - sY, 0.0) * np.sign(s[0])]

# (i) relaxation from sigma0 = 400 MPa at fixed strain
# exact solution (4.19): sigma - sY = (sigma0 - sY) exp(-t/tau), here at t = 5 tau
sol = solve_ivp(rhs, [0, 5 * tau], [400.0], args=(0.0,), rtol=1e-10, atol=1e-10)
print("relaxation  numeric %.4f  exact %.4f" % (sol.y[0, -1], sY + 150 * np.exp(-5)))
# (ii) constant strain rate c from the virgin state: overstress -> eta*c
# integrate long enough to reach yield (3 sY/(E c)) and then the steady state (20 tau)
for c in [1e-4, 1e-3, 1e-2]:
    sol = solve_ivp(rhs, [0, 20 * tau + 3 * sY / (E * c)], [0.0], args=(c,),
                    rtol=1e-10, atol=1e-10)
    print("rate %.0e  sigma_inf numeric %.4f  exact %.4f" % (c, sol.y[0, -1], sY + eta * c))
# relaxation  numeric 251.0107  exact 251.0107
# rate 1e-04  sigma_inf numeric 250.1000  exact 250.1000
# rate 1e-03  sigma_inf numeric 251.0000  exact 251.0000
# rate 1e-02  sigma_inf numeric 260.0000  exact 260.0000
