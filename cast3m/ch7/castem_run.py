"""castem_run.py -- Cast3M as a black-box direct solver for the drivers of the
companion exercises C7.3-C7.6: fill the template, run Cast3M in its own
directory, read the CSV it writes, keep a cache and a counter of the runs."""
import os
import re
import shlex
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# Command that runs Cast3M on a file: castem26 by default; for tests without
# Cast3M, CASTEM="python3 <this dir>/tools/castem_stub.py" (see the README).
CASTEM = os.environ.get("CASTEM", "castem26")
TEMPLATE = (HERE / "truss.template").read_text()
RUNS = HERE / "runs"                     # one subdirectory per computation

P_TRUE = np.array([200e3, 200.0, 10e3])  # E, sY, H in MPa (Exercise 7.9)
UREF = P_TRUE[1] / P_TRUE[0]             # displacement scale sY l / E, l = 1 mm

nruns = 0                                # number of Cast3M runs so far
_cache = {}
_lock = threading.Lock()


def read_csv(path):
    """Columns of a CSV file written by EXPORTCSV (header ABSC;UX;UY)."""
    lines = Path(path).read_text().split()
    head = lines[0].split(";")
    data = np.array([[float(x) for x in l.split(";")] for l in lines[1:]])
    return {h: data[:, i] for i, h in enumerate(head)}


def displacements(p):
    """u(t_n) of node O, shape (81, 2), for p = (E, sY, H) in MPa: one Cast3M run,
    cached on the 12 significant digits of p (finite differences reuse runs)."""
    global nruns
    key = tuple(float("%.12g" % x) for x in p)
    if key in _cache:
        return _cache[key]
    tag = "E%.10g_sY%.10g_H%.10g" % key
    wd = RUNS / re.sub(r"[^0-9A-Za-z_.+-]", "_", tag)
    wd.mkdir(parents=True, exist_ok=True)
    text = (TEMPLATE.replace("__E__", "%.12e" % key[0])
                    .replace("__SY__", "%.12e" % key[1])
                    .replace("__H__", "%.12e" % key[2]))
    (wd / "calcul.dgibi").write_text(text)
    # each run in its own directory: Cast3M writes fort.* files where it runs
    with open(wd / "sortie.log", "w") as log:
        subprocess.run(shlex.split(CASTEM) + ["calcul.dgibi"], cwd=wd,
                       stdout=log, stderr=subprocess.STDOUT)
    with _lock:
        nruns += 1
    # Cast3M returns 0 even when the computation fails: look at the output
    out = (wd / "sortie.log").read_text(errors="replace")
    if "ERREUR" in out or not (wd / "truss_u.csv").exists():
        raise RuntimeError("Cast3M failed in %s, see sortie.log" % wd)
    c = read_csv(wd / "truss_u.csv")
    u = np.column_stack([c["UX"], c["UY"]])
    _cache[key] = u
    return u


def jacobian_fd(p, rel=1e-4, workers=6):
    """du/dlog p by central differences, shape (81, 2, 3): six runs, in parallel.
    The step rel balances truncation (rel^2) against the solver tolerance / rel."""
    p = np.asarray(p, float)
    pts = []
    for j in range(3):
        for s in (+1, -1):
            q = p.copy()
            q[j] *= np.exp(s * rel)
            pts.append(q)
    with ThreadPoolExecutor(workers) as ex:
        us = list(ex.map(displacements, pts))
    return np.stack([(us[2 * j] - us[2 * j + 1]) / (2 * rel) for j in range(3)], axis=-1)


def measurements():
    """Measured displacements (81, 2) of truss_measurements.csv (make_measurements.py)."""
    c = read_csv(HERE / "truss_measurements.csv")
    return np.column_stack([c["UX"], c["UY"]])


def clean():
    """Remove the run directories."""
    shutil.rmtree(RUNS, ignore_errors=True)
