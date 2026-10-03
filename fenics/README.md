# FEniCSx scripts

| file | chapter | status |
|---|---|---|
| `cylinder_plasticity.py` | Ch. 5, thick-walled cylinder, radial return + global Newton (consistent or continuum tangent) | **untested** — written for dolfinx 0.8/0.9, not run |

The return map is imported from `python/examples/ch5/ci_core.py`; FEniCSx only
assembles the residual and the tangent from quadrature-point fields. Expected
values (Python reference): 47 Newton iterations with the consistent tangent,
72 with the continuum one, `u(a) = 3.4526 mm`.

Run (serial): `cd fenics && python3 cylinder_plasticity.py`.
Files marked untested are not printed in the book until they have been run.
