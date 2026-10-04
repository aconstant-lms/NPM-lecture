"""castem_stub.py -- stands in for Cast3M when testing the drivers: reads the
parameters of calcul.dgibi written from truss.template and writes the file
truss_u.csv that the Cast3M run would write, computed with id_core (Chapter 7)."""
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "python" / "examples" / "ch7"))
from id_core import Truss  # noqa: E402

text = Path(sys.argv[1]).read_text()
val = {k: float(re.search(r"^\s*%s\s*=\s*([-+0-9.eE]+)\s*;" % k, text, re.M).group(1))
       for k in ("yo", "sy", "hh")}
p = np.array([val["yo"], val["sy"], val["hh"]])
# same loading as truss.template: N0 = 200 N, fixed (not the parameter sY)
Q = lambda t: np.array([2.3 * 200.0 * np.sin(t), 0.8 * 200.0 * np.sin(2 * t)])
ts = np.linspace(0, 2 * np.pi, 81)
u = Truss(Q, ts).forward(p)[0]
with open("truss_u.csv", "w") as f:
    f.write("ABSC;UX;UY\n")
    for t, (ux, uy) in zip(ts, u):
        f.write("%.17g;%.17g;%.17g\n" % (t, ux, uy))
print("castem_stub: E = %g, sY = %g, H = %g, fin normale" % tuple(p))
