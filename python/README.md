# Python files of the lecture notes

    figures/ch1/   scripts producing the figures of Chapter 1 (output: ../../../figures/ch1/)
    figures/appB/  script producing the Gateaux/Frechet figure of Appendix B
    examples/ch1/  computational exercises of Chapter 1 and the constrained spring chain
    examples/ch2/  computational exercises of Chapter 2
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
