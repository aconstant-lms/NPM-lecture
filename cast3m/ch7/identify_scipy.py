"""identify_scipy.py -- companion exercise C7.4: identification of (E, sY, H) on
the three-bar truss with Cast3M as the direct solver and SciPy's least_squares
as the driver; Jacobian by central differences in log p (six parallel runs)."""
import time

import numpy as np
from scipy.optimize import least_squares

import castem_run as cr

# Exercise 7.9 with Cast3M: the measurements are those of the book (seed 0),
# the parameters q = log(p / p0), the residuals (u_n(p) - u^m_n) / u_ref for the
# 81 instants and both components, J = |r|^2 / 2 (the cost of id_core.Truss).
um = cr.measurements()


def residuals(q, p0):
    return ((cr.displacements(p0 * np.exp(q)) - um) / cr.UREF).ravel()


def jacobian(q, p0):                    # (162 x 3): d r / d q by central differences
    return (cr.jacobian_fd(p0 * np.exp(q)) / cr.UREF).reshape(-1, 3)


for start in ((0.7, 1.2, 3.0), (0.7, 1.5, 3.0)):
    p0 = cr.P_TRUE * np.array(start)
    cr.nruns, t0 = 0, time.time()
    res = least_squares(residuals, np.zeros(3), jac=jacobian, args=(p0,),
                        method="lm", xtol=1e-10, ftol=1e-12)
    p = p0 * np.exp(res.x)
    print("start %s: p/p_true = %s, J = %.4f, %d Cast3M runs, %.0f s"
          % (start, np.round(p / cr.P_TRUE, 4), res.cost, cr.nruns, time.time() - t0))
# Expected (Exercise 7.9, exact derivatives): from (0.7, 1.2, 3) the minimum
# p/p_true = (1.0022, 0.9992, 1.0036), J = 0.2911; from (0.7, 1.5, 3) the truss
# never yields at the start, sY and H receive no gradient, and the iteration
# ends at E/E_true = 0.4424 with J = 522.5.
