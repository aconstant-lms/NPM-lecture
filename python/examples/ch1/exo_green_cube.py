"""Exercise green-cube (ch1) of the lecture notes "Nonlinear Problems in Mechanics".
Run: python3 python/examples/ch1/exo_green_cube.py
"""
# Exercise 1.14: Green's first identity on the unit cube, computed exactly,
#   int grad u . grad v dV = - int u Laplacian(v) dV + int u dv/dn dS,
# for u = x y and v = y z. Here Laplacian(v) = 0: only the boundary term remains.
import sympy as sp
x, y, z = sp.symbols('x y z'); u, v = x*y, y*z
# Left-hand side: volume integral of grad u . grad v = x z.
lhs = sp.integrate(sum(sp.diff(u,s)*sp.diff(v,s) for s in (x,y,z)),
                   (x,0,1),(y,0,1),(z,0,1))
# The six faces: (variable w fixed, its value, sign of the outward normal n,
# the two variables of the face). On the face w = val, dv/dn = sign * dv/dw.
faces = [(x,0,-1,(y,z)),(x,1,1,(y,z)),(y,0,-1,(x,z)),
         (y,1,1,(x,z)),(z,0,-1,(x,y)),(z,1,1,(x,y))]
# Boundary term: sum over the faces of int u dv/dn.
bnd = sum(sp.integrate((u*sg*sp.diff(v,w)).subs(w,val),(a,0,1),(b,0,1))
          for w,val,sg,(a,b) in faces)
print(lhs, bnd)        # 1/4 1/4
