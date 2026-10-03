"""Exercise green-cube (ch1) of the lecture notes "Nonlinear Problems in Mechanics".
Run: python3 python/examples/ch1/exo_green_cube.py
"""
import sympy as sp
x, y, z = sp.symbols('x y z'); u, v = x*y, y*z
lhs = sp.integrate(sum(sp.diff(u,s)*sp.diff(v,s) for s in (x,y,z)),
                   (x,0,1),(y,0,1),(z,0,1))
faces = [(x,0,-1,(y,z)),(x,1,1,(y,z)),(y,0,-1,(x,z)),
         (y,1,1,(x,z)),(z,0,-1,(x,y)),(z,1,1,(x,y))]
bnd = sum(sp.integrate((u*sg*sp.diff(v,w)).subs(w,val),(a,0,1),(b,0,1))
          for w,val,sg,(a,b) in faces)
print(lhs, bnd)        # 1/4 1/4
