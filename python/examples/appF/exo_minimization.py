"""Exercise F.x -- the Rosenbrock function written as a least-squares problem,
r = (10 (x2 - x1^2), 1 - x1), J = |r|^2 / 2, minimized from (-1.2, 1) by six
methods: steepest descent, Nelder-Mead, BFGS, Newton, Gauss-Newton and
Levenberg-Marquardt. Counts of evaluations to reach |x - x*| < 1e-6."""
import numpy as np
from scipy.optimize import minimize, least_squares

r = lambda x: np.array([10 * (x[1] - x[0] ** 2), 1 - x[0]])
Jr = lambda x: np.array([[-20 * x[0], 10.0], [-1.0, 0.0]])        # dr/dx
J = lambda x: 0.5 * r(x) @ r(x)
g = lambda x: Jr(x).T @ r(x)                                        # gradient
H = lambda x: Jr(x).T @ Jr(x) + r(x)[0] * np.array([[-20.0, 0], [0, 0]])
x0, xs = np.array([-1.2, 1.0]), np.array([1.0, 1.0])


def armijo(x, d, f0, g0):
    """Backtracking line search: halve the step until sufficient decrease."""
    a, n = 1.0, 0
    while J(x + a * d) > f0 + 1e-4 * a * g0 @ d:
        a, n = a / 2, n + 1
    return a, n + 1


def descent(direction, kmax=100000):
    x, nf = x0.copy(), 0
    for k in range(kmax):
        if np.linalg.norm(x - xs) < 1e-6:
            return k, nf
        d = direction(x)
        a, n = armijo(x, d, J(x), g(x))
        x, nf = x + a * d, nf + n
    return kmax, nf


k, nf = descent(lambda x: -g(x))
print("steepest descent + Armijo : %6d iterations, %6d values" % (k, nf))
k, nf = descent(lambda x: -np.linalg.solve(H(x), g(x)))
print("Newton + Armijo           : %6d iterations, %6d values" % (k, nf))
k, nf = descent(lambda x: -np.linalg.lstsq(Jr(x), r(x), rcond=None)[0])
print("Gauss-Newton + Armijo     : %6d iterations, %6d values" % (k, nf))
res = minimize(J, x0, method="Nelder-Mead", options=dict(xatol=1e-8, fatol=1e-16))
print("Nelder-Mead               : %6d iterations, %6d values, error %.1e" %
      (res.nit, res.nfev, np.linalg.norm(res.x - xs)))
res = minimize(J, x0, jac=g, method="BFGS", options=dict(gtol=1e-10))
print("BFGS                      : %6d iterations, %6d values+gradients, error %.1e" %
      (res.nit, res.nfev, np.linalg.norm(res.x - xs)))
res = least_squares(r, x0, jac=Jr, method="lm", xtol=1e-12)
print("Levenberg-Marquardt       : %6d residuals, %6d Jacobians, error %.1e" %
      (res.nfev, res.njev, np.linalg.norm(res.x - xs)))
