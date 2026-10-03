"""Exercise bsplines (ch1) of the lecture notes "Nonlinear Problems in Mechanics".
Run: python3 python/examples/ch1/exo_bsplines.py
"""
import sympy as sp
xi, t = sp.symbols("xi t")
Xi = [0, 0, 0, 1, 2, 3, 3, 3]                    # knot vector, p = 2
spans = [(0, 1), (1, 2), (2, 3)]                 # the three elements

def N(i, p, a, b):
    """Polynomial piece of N_{i,p} on the span [a,b) (Cox-de Boor, 0/0 = 0)."""
    if p == 0:
        return sp.Integer(1 if (Xi[i], Xi[i+1]) == (a, b) else 0)
    d1, d2 = Xi[i+p] - Xi[i], Xi[i+p+1] - Xi[i+1]
    r = 0
    if d1: r += (xi - Xi[i])/d1*N(i, p-1, a, b)
    if d2: r += (Xi[i+p+1] - xi)/d2*N(i+1, p-1, a, b)
    return sp.expand(r)

n = len(Xi) - 2 - 1                              # 5 functions
pc = [[N(i, 2, a, b) for (a, b) in spans] for i in range(n)]
for i, P in enumerate(pc):
    print(f"N_{i+1}:", [sp.factor(q) for q in P])
print("sum on each span:", [sp.simplify(sum(P[e] for P in pc)) for e in range(3)])
for e, x0 in [(0, 1), (1, 2)]:                   # interior knots 1 and 2
    jump = lambda q, m: sp.diff(q[e+1] - q[e], xi, m).subs(xi, x0)
    print(f"jumps of N, N', N'' at xi={x0}:",
          [[jump(P, m) for m in range(3)] for P in pc])
# Bezier extraction on element 2: N^e(t) = C^e b(t), t = xi - 1 in [0,1]
b = [(1-t)**2, 2*t*(1-t), t**2]                  # Bernstein polynomials
Ne = [sp.expand(P[1].subs(xi, t + 1)) for P in pc if P[1] != 0]   # N_2, N_3, N_4
mono = lambda f: [sp.Poly(f, t).coeff_monomial(t**k) for k in range(3)]
Ce = sp.Matrix([mono(f) for f in Ne]) * sp.Matrix([mono(f) for f in b]).inv()
print("C^e (element 2) =", Ce.tolist())
# N_1: [(xi - 1)**2, 0, 0]   N_2: [-xi*(3*xi - 4)/2, (xi - 2)**2/2, 0]
# N_3: [xi**2/2, -(2*xi**2 - 6*xi + 3)/2, (xi - 3)**2/2]
# N_4: [0, (xi - 1)**2/2, -(xi - 3)*(3*xi - 5)/2]   N_5: [0, 0, (xi - 2)**2]
# sum on each span: [1, 1, 1]
# jumps at xi=1: N and N' jump by 0; N'' jumps by -2, 4, -3, 1, 0 (C^1, not C^2)
# C^e (element 2) = [[1/2, 0, 0], [1/2, 1, 1/2], [0, 0, 1/2]]
