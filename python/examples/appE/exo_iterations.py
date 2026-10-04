"""Exercise "Five iterations on two test problems" (Appendix E).
Run: cd python/examples/appE && python3 exo_iterations.py
"""
# Exercise E.3: Picard, averaged (Krasnoselskii-Mann) and Anderson iterations for
# a fixed point x = g(x), Newton and modified Newton for F(x) = 0, on x = cos x
# (a contraction, Section E.1) and on a rotation (non-expansive, Section E.2).
# All stop when the increment is below tol = 1e-12; they return (x, iterations).
import numpy as np


def picard(g, x, tol=1e-12, kmax=500):
    """Picard iteration x_{k+1} = g(x_k) (Section E.1)."""
    for k in range(kmax):
        x_new = g(x)
        if np.max(np.abs(x_new - x)) < tol:
            return x_new, k + 1
        x = x_new
    return x, kmax


def km(g, x, theta=0.5, tol=1e-12, kmax=500):        # Krasnoselskii--Mann
    """Averaged iteration x_{k+1} = (1 - theta) x_k + theta g(x_k), (E.3):
    Picard on the averaged map (Section E.2)."""
    return picard(lambda y: (1 - theta) * y + theta * g(y), x, tol, kmax)


def anderson(g, x, m=3, tol=1e-12, kmax=500):
    """Anderson acceleration with m differences, (E.6) in Section E.4: keep the
    last m+1 values g_i = g(x_i) and residuals f_i = g_i - x_i, then
    gamma = argmin |f_k - dF gamma|,  x_{k+1} = g_k - dG gamma."""
    x = np.atleast_1d(np.asarray(x, float))
    G, F = [], []
    for k in range(kmax):
        gx = np.atleast_1d(g(x))
        f = gx - x
        # Here the test is on the residual f = g(x) - x, the next Picard increment.
        if np.max(np.abs(f)) < tol:
            return x, k + 1
        # Keep the m+1 most recent values and residuals.
        G, F = (G + [gx])[-(m + 1):], (F + [f])[-(m + 1):]
        if len(F) == 1:
            x = gx
        else:
            # Columns: the differences f_{i+1} - f_i and g_{i+1} - g_i.
            dF = np.array([F[i + 1] - F[i] for i in range(len(F) - 1)]).T
            dG = np.array([G[i + 1] - G[i] for i in range(len(G) - 1)]).T
            x = gx - dG @ np.linalg.lstsq(dF, f, rcond=None)[0]
    return x, kmax


def newton(F, J, x, tol=1e-12, kmax=50, frozen=False):
    """Newton's method J(x_k) dx = -F(x_k), x_{k+1} = x_k + dx, (E.4) in
    Section E.3. With frozen=True, modified Newton: the Jacobian J(x_0) is kept."""
    J0 = J(x)
    for k in range(kmax):
        dx = np.linalg.solve(np.atleast_2d(J0 if frozen else J(x)), np.atleast_1d(-F(x)))
        x = x + dx
        if np.max(np.abs(dx)) < tol:
            return x, k + 1
    return x, kmax


# 1. a contraction: x = cos x
# The rate of Picard is |g'(x*)| = sin(x*) = 0.674 (Exercise E.1).
g = np.cos
x0 = np.array([1.0])
print("x = cos x, fixed point", picard(g, x0)[0][0], "|g'(x*)| =",
      abs(np.sin(picard(g, x0)[0][0])))
print("  Picard   :", picard(g, x0)[1], "iterations")
print("  KM 1/2   :", km(g, x0)[1])
print("  Anderson :", anderson(g, x0, m=1)[1])
# Newton on F(x) = x - cos x, with Jacobian F'(x) = 1 + sin x.
F, J = (lambda x: x - np.cos(x)), (lambda x: np.atleast_2d(1 + np.sin(x)))
print("  Newton   :", newton(F, J, x0)[1], "| modified Newton:", newton(F, J, x0, frozen=True)[1])

# 2. a non-expansive map: rotation by 90 degrees, fixed point 0
# Picard cycles with period four (Exercise E.2); averaging with theta = 1/2 is a
# contraction of factor sqrt(2)/2; Anderson with m = 2 solves the linear problem.
R = np.array([[0.0, -1.0], [1.0, 0.0]])
rot = lambda x: R @ x
x0 = np.array([1.0, 0.0])
xp, kp = picard(rot, x0, kmax=200)
print("rotation: Picard after", kp, "iterations at", xp, "(cycles, never converges)")
print("  KM 1/2   :", km(rot, x0)[1], "iterations")
print("  Anderson :", anderson(rot, x0, m=2)[1], "iterations")
# x = cos x, fixed point 0.7390851332147726 |g'(x*)| = 0.6736120291829281
#   Picard   : 69 iterations
#   KM 1/2   : 16
#   Anderson : 7
#   Newton   : 5 | modified Newton: 12
# rotation: Picard after 200 iterations at [1. 0.] (cycles, never converges)
#   KM 1/2   : 79 iterations
#   Anderson : 4 iterations
