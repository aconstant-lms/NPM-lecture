"""Exercise "The Tresca corner" (ch5) of the lecture notes.
Run: cd python/examples/ch5 && python3 exo_tresca.py
"""
import numpy as np
from ci_core import tresca_return
E, nu, sY = 200e3, 0.3, 250.0
mu = E / (2 * (1 + nu))
for f1, f2 in [(20, 20), (20, 11), (20, 10), (20, 9), (20, 5)]:
    s_tr = np.array([sY + f1, sY + f2, 0.0])   # f1 = s1 - s3 - sY, f2 = s2 - s3 - sY
    s, mode, dg = tresca_return(s_tr, mu, sY)
    print(f"f1 = {f1}, f2 = {f2:2d}: {mode:6s} dg = ({dg[0]:.4e}, {dg[1]:.4e}),"
          f" s = {np.round(s, 4)}, max shear check {s.max() - s.min():.4f}")
# f1 = 20, f2 = 20: corner dg = (4.3333e-05, 4.3333e-05), s = [263.3333 263.3333  13.3333], max shear check 250.0000
# f1 = 20, f2 = 11: corner dg = (6.2833e-05, 4.3333e-06), s = [260.3333 260.3333  10.3333], max shear check 250.0000
# f1 = 20, f2 = 10: face   dg = (6.5000e-05, 0.0000e+00), s = [260. 260.  10.], max shear check 250.0000
# f1 = 20, f2 =  9: face   dg = (6.5000e-05, 0.0000e+00), s = [260. 259.  10.], max shear check 250.0000
# f1 = 20, f2 =  5: face   dg = (6.5000e-05, 0.0000e+00), s = [260. 255.  10.], max shear check 250.0000
