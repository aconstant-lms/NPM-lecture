"""Spring chain with an imposed displacement (Chapter 1, constraints and
Lagrange multipliers). Lagrangian L = 1/2 u^T K u - f^T u - lambda (P u - u_D),
so that lambda is the reaction. Solves the symmetric saddle-point system
    [ K  -P^T] [u     ]   [ f  ]
    [-P   0  ] [lambda] = [-u_D]
and prints the displacements, the multiplier and the reaction r = P^T lambda.
Run: python3 python/examples/ch1/springs_constraint.py
"""
import numpy as np

B = np.array([[1, 0, 0], [-1, 1, 0], [0, -1, 1], [0, 0, -1]], dtype=float)
C = np.diag([1.0, 1.0, 1.0, 1.0])          # spring stiffnesses c_i > 0
K = B.T @ C @ B
f = np.zeros(3)                             # no external force
P = np.array([[0.0, 0.0, 1.0]])             # constraint u_3 = u_D
uD = np.array([1.0])

A = np.block([[K, -P.T], [-P, np.zeros((1, 1))]])
sol = np.linalg.solve(A, np.concatenate([f, -uD]))
u, lam = sol[:3], sol[3:]
r = P.T @ lam
print("u      =", np.round(u, 4))               # [0.3333 0.6667 1.    ]
print("lambda =", np.round(lam, 4))             # [1.3333]
print("r      =", np.round(r, 4))               # [0.     0.     1.3333]
print("check  B^T sigma = f + r:", np.allclose(B.T @ (C @ (B @ u)), f + r))  # True
