"""Exercise neumann (ch1) of the lecture notes "Nonlinear Problems in Mechanics".
Run: python3 python/examples/ch1/exo_neumann.py
"""
# Exercise 1.18, Ritz check: pure Neumann problem with f = 1 and g = -1,
#   E(u) = int_0^1 (1/2 u'^2 - f u) dx - g u(1),
# minimized over the quadratics u = a0 + a1 x + a2 x^2.
import sympy as sp
x, a0, a1, a2 = sp.symbols('x a0 a1 a2')
u = a0 + a1*x + a2*x**2
# Energy with f = 1 and g = -1 (the term - g u(1) becomes + u(1)).
E = sp.integrate(sp.diff(u,x)**2/2 - u, (x,0,1)) + u.subs(x,1)
# dE/da0 vanishes identically: E does not see constants (solution up to a constant).
print(sp.diff(E, a0))                                    # 0: a0 free
# The other two stationarity conditions give u' = -x, the exact solution.
print(sp.solve([sp.diff(E,a1), sp.diff(E,a2)], [a1,a2])) # a1=0, a2=-1/2
