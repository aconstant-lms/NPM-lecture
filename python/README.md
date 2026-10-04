# Python files of the lecture notes

    figures/ch1/   scripts producing the figures of Chapter 1 (output: ../../../figures/ch1/)
    figures/appB/  script producing the Gateaux/Frechet figure of the appendix on
                   variational derivatives (file appB_..., printed as Appendix C)
    figures/ch2/   figures of Chapter 2 (Ritz bar, conditioning, hourglass, reference
                   map, Kirsch plate, directional Young modulus);
                   style_ch2.py is their common style
    examples/ch1/  computational exercises of Chapter 1 (incl. B-splines exo_bsplines.py)
                   and the constrained spring chain
    examples/ch2/  computational exercises of Chapter 2: elastic_constants.py (stiffness
                   and compliance tables), exo_conditioning.py, and the Kirsch plate with
                   a hole in three versions: kirsch_fem.py (printed), kirsch_fenics.py
                   (FEniCS/dolfinx) and kirsch_cast3m.dgibi (Cast3M); the last two are
                   only in the repository, not quoted in the text
    figures/ch4/   figures of Chapter 4 (filament cycles, yield loci, hardening surfaces)
    examples/ch4/  computational exercises of Chapter 4 (1D cycles, viscoplasticity,
                   nonlinear hardening, plane-strain sheet)
    figures/ch5/   figures of Chapter 5 (forward/backward Euler, iso-error map, Newton)
    examples/ch5/  return-map library ci_core.py (Boxes 5.1-5.5), axisymmetric FE
                   solver ci_fem.py, and the exercise scripts of Chapter 5
                   (run them from python/examples/ch5: they import ci_core)

The exercise files are printed in the solutions with `\lstinputlisting`,
so editing a file here changes the printed solution. Each file runs on its own:

    pip install numpy scipy sympy matplotlib
    python3 python/examples/ch1/exo_gd.py
    cd python/figures/ch1 && python3 fig_resonance.py
