"""make_measurements.py -- the synthetic measurements of Exercise 7.9 (same model,
seed and noise as python/examples/ch7/exo_truss_id.py), written as the CSV file
truss_measurements.csv read by the drivers of the companion exercises C7.3-C7.6."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "python" / "examples" / "ch7"))
from id_core import Truss  # noqa: E402

E, sY, H = 200e3, 200.0, 10e3                      # MPa
p_true = np.array([E, sY, H])
Q = lambda t: np.array([2.3 * sY * np.sin(t), 0.8 * sY * np.sin(2 * t)])
ts = np.linspace(0, 2 * np.pi, 81)                 # one period, 80 steps
T = Truss(Q, ts)
uref = sY / E
u_true = T.forward(p_true)[0]
rng = np.random.default_rng(0)                     # noise: 1% of max |u|
um = u_true + 0.01 * np.abs(u_true).max() * rng.standard_normal(u_true.shape)

with open(HERE / "truss_measurements.csv", "w") as f:
    f.write("ABSC;UX;UY\n")
    for t, (ux, uy) in zip(ts, um):
        f.write("%.17g;%.17g;%.17g\n" % (t, ux, uy))
print("J at p_true:", T.cost(p_true, um, uref))
# J at p_true: 0.2974316186609882
