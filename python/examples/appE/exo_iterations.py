"""Exercise "Five iterations on two test problems" (Appendix E).
Run: cd python/examples/appE && python3 exo_iterations.py
"""
import numpy as np


def picard(g, x, tol=1e-12, kmax=500):
    for k in range(kmax):
        x_new = g(x)
        if np.max(np.abs(x_new - x)) < tol:
            return x_new, k + 1
        x = x_new
    return x, kmax


def km(g, x, theta=0.5, tol=1e-12, kmax=500):        # Krasnoselskii--Mann
    return picard(lambda y: (1 - theta) * y + theta * g(y), x, tol, kmax)


def anderson(g, x, m=3, tol=1e-12, kmax=500):
    x = np.atleast_1d(np.asarray(x, float))
    G, F = [], []
    for k in range(kmax):
        gx = np.atleast_1d(g(x))
        f = gx - x
        if np.max(np.abs(f)) < tol:
            return x, k + 1
        G, F = (G + [gx])[-(m + 1):], (F + [f])[-(m + 1):]
        if len(F) == 1:
            x = gx
        else:
            dF = np.array([F[i + 1] - F[i] for i in range(len(F) - 1)]).T
            dG = np.array([G[i + 1] - G[i] for i in range(len(G) - 1)]).T
            x = gx - dG @ np.linalg.lstsq(dF, f, rcond=None)[0]
    return x, kmax


def newton(F, J, x, tol=1e-12, kmax=50, frozen=False):
    J0 = J(x)
    for k in range(kmax):
        dx = np.linalg.solve(np.atleast_2d(J0 if frozen else J(x)), np.atleast_1d(-F(x)))
        x = x + dx
        if np.max(np.abs(dx)) < tol:
            return x, k + 1
    return x, kmax


# 1. a contraction: x = cos x
g = np.cos
x0 = np.array([1.0])
print("x = cos x, fixed point", picard(g, x0)[0][0], "|g'(x*)| =",
      abs(np.sin(picard(g, x0)[0][0])))
print("  Picard   :", picard(g, x0)[1], "iterations")
print("  KM 1/2   :", km(g, x0)[1])
print("  Anderson :", anderson(g, x0, m=1)[1])
F, J = (lambda x: x - np.cos(x)), (lambda x: np.atleast_2d(1 + np.sin(x)))
print("  Newton   :", newton(F, J, x0)[1], "| modified Newton:", newton(F, J, x0, frozen=True)[1])

# 2. a non-expansive map: rotation by 90 degrees, fixed point 0
R = np.array([[0.0, -1.0], [1.0, 0.0]])
rot = lambda x: R @ x
x0 = np.array([1.0, 0.0])
xp, kp = picard(rot, x0, kmax=200)
print("rotation: Picard after", kp, "iterations at", xp, "(cycles, never converges)")
print("  KM 1/2   :", km(rot, x0)[1], "iterations")
print("  Anderson :", anderson(rot, x0, m=2)[1], "iterations")
