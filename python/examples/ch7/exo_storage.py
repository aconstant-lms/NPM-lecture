"""Exercise 7.x -- what the adjoint must store: the truss gradient with
checkpoints (exact recomputation of the steps between them) or with the states
interpolated linearly in time between checkpoints; distinct tangent matrices."""
import numpy as np
from id_core import Truss, adjoint_with_storage, active_sets

# Exercise 7.12, Section 7.6.4: the truss of Exercise 7.9 (same data and noise);
# the gradient is computed at p1 = (0.9, 1.2, 1.5) p_true. Moduli in MPa.
E, sY, H = 200e3, 200.0, 10e3
p_true = np.array([E, sY, H])
Q = lambda t: np.array([2.3 * sY * np.sin(t), 0.8 * sY * np.sin(2 * t)])
ts = np.linspace(0, 2 * np.pi, 81)
T = Truss(Q, ts)
uref = sY / E
u_true = T.forward(p_true)[0]
rng = np.random.default_rng(0)
um = u_true + 0.01 * np.abs(u_true).max() * rng.standard_normal(u_true.shape)
p1 = p_true * np.array([0.9, 1.2, 1.5])
g_all = T.gradient_adjoint(p1, um, uref)            # every step stored

# Table 7.6: c = checkpoint spacing; states held = N/c checkpoints + one segment
# of c - 1 recomputed states; errors relative to g_all (max norm)

print("  c  checkpoints  peak states (recompute)  extra steps  error (recompute)"
      "  error (interpolate)")
for c in [1, 2, 4, 5, 8, 10, 20, 40]:
    g_r, n_ck, extra = adjoint_with_storage(T, p1, um, uref, c, "recompute")
    g_i = adjoint_with_storage(T, p1, um, uref, c, "interpolate")[0]
    err = lambda g: np.abs(g - g_all).max() / np.abs(g_all).max()
    print("%3d  %11d  %23d  %11d  %17.1e  %19.1e" %
          (c, n_ck, n_ck + (c - 1 if c > 1 else 0), extra if c > 1 else 0, err(g_r), err(g_i)))

# plastic patterns (one bit per bar) along the 80 steps: with linear hardening
# each pattern gives one consistent tangent ("Piecewise-constant tangents")
for lab, p in [("p1", p1), ("p_true", p_true)]:
    pats, distinct = active_sets(T, p)
    changes = sum(pats[n] != pats[n - 1] for n in range(1, len(pats)))
    print("%s: %d plastic steps, %d changes of the plastic pattern, distinct "
          "tangents %s" % (lab, sum(any(q) for q in pats), changes,
                           [tuple(int(b) for b in q) for q in distinct]))
