"""Spring chain with an imposed displacement (Chapter 1, constraints and
Lagrange multipliers). Lagrangian L = 1/2 u^T K u - f^T u - lambda (P u - u_D),
so that lambda is the reaction. Solves the symmetric saddle-point system
    [ K  -P^T] [u     ]   [ f  ]
    [-P   0  ] [lambda] = [-u_D]
and prints the displacements, the multiplier and the reaction r = P^T lambda.
Run: python3 python/examples/ch1/springs_constraint.py

This is the example "an imposed displacement in the spring chain" of Section 1.2.8,
system (1.6), with unit stiffnesses, f = 0 and u_D = 1 (Figure 1.7).
"""
import numpy as np

# Chain of three masses and four springs between two walls (Section 1.2.2):
# elongations e = B u, tensions sigma = C e, stiffness K = B^T C B.
B = np.array([[1, 0, 0], [-1, 1, 0], [0, -1, 1], [0, 0, -1]], dtype=float)
C = np.diag([1.0, 1.0, 1.0, 1.0])          # spring stiffnesses c_i > 0
K = B.T @ C @ B
f = np.zeros(3)                             # no external force
# Essential condition P u = u_D on the third mass.
P = np.array([[0.0, 0.0, 1.0]])             # constraint u_3 = u_D
uD = np.array([1.0])

# Stationarity of L: K u - f - P^T lambda = 0 and P u = u_D, written as the
# symmetric saddle-point system (1.6).
A = np.block([[K, -P.T], [-P, np.zeros((1, 1))]])
sol = np.linalg.solve(A, np.concatenate([f, -uD]))
u, lam = sol[:3], sol[3:]
# Reaction of the support: r = P^T lambda (positive: it pushes along x).
r = P.T @ lam
print("u      =", np.round(u, 4))               # [0.3333 0.6667 1.    ]
print("lambda =", np.round(lam, 4))             # [1.3333]
print("r      =", np.round(r, 4))               # [0.     0.     1.3333]
# Equilibrium of the masses with the reaction: B^T sigma = f + r.
print("check  B^T sigma = f + r:", np.allclose(B.T @ (C @ (B @ u)), f + r))  # True
