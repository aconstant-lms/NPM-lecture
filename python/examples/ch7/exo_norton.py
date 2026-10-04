"""Exercise 7.x -- Norton-Hoff filament in a relaxation test: identifiability of
(E, sY, K, m), Gauss-Newton eigenvalues against the observation window, and
Levenberg-Marquardt identifications from several starts."""
import numpy as np
from scipy.optimize import least_squares
from id_core import norton_relax, relaxation_test, gauss_newton

p_true = np.array([200e3, 200.0, 500.0, 5.0])     # E, sY (MPa), K (MPa s^1/m), m
sref = 200.0
rng = np.random.default_rng(1)
np.seterr(all="ignore")       # Levenberg-Marquardt tries some extreme trial points

for t_hold in [10.0, 100.0, 1000.0]:
    tt, ee = relaxation_test(t_hold=t_hold)
    s, ds = norton_relax(p_true, tt, ee)
    G, lam, V, C = gauss_newton(ds[1:] / sref, scale=p_true)
    noise = 0.01 * np.abs(s).max() / sref
    print("hold %6.0f s: GN eigenvalues %s, condition %.1e" %
          (t_hold, np.array2string(lam, precision=3), lam[-1] / lam[0]))
    print("   weakest direction (E, sY, K, m) %s; std along it %.3f (1%% noise)" %
          (np.round(V[:, 0], 2), noise / np.sqrt(lam[0])))
    print("   correlations K-m %.3f, sY-K %.3f, sY-m %.3f" % (C[2, 3], C[1, 2], C[1, 3]))

    sm = s + noise * sref * rng.standard_normal(s.shape)
    resid = lambda q: (norton_relax(p_true * np.exp(q), tt, ee, False) - sm)[1:] / sref
    jac = lambda q: norton_relax(p_true * np.exp(q), tt, ee)[1][1:] / sref * p_true * np.exp(q)
    for f0 in ([1.3, 0.5, 2.0, 0.6], [0.7, 1.5, 0.5, 1.6], [1.2, 1.2, 1.2, 1.2]):
        q0 = np.log(np.array(f0))
        r = least_squares(resid, q0, jac=jac, method="lm", xtol=1e-12, ftol=1e-14)
        print("   start %s -> p/p_true %s, rms misfit %.4f MPa, %d iterations" %
              (f0, np.round(np.exp(r.x), 3), sref * np.sqrt(np.mean(r.fun ** 2)), r.njev))
