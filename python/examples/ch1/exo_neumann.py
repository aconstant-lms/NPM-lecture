"""Exercise neumann (ch1) of the lecture notes "Nonlinear Problems in Mechanics".
Run: python3 python/examples/ch1/exo_neumann.py
"""
import sympy as sp
x, a0, a1, a2 = sp.symbols('x a0 a1 a2')
u = a0 + a1*x + a2*x**2
E = sp.integrate(sp.diff(u,x)**2/2 - u, (x,0,1)) + u.subs(x,1)
print(sp.diff(E, a0))                                    # 0: a0 free
print(sp.solve([sp.diff(E,a1), sp.diff(E,a2)], [a1,a2])) # a1=0, a2=-1/2
