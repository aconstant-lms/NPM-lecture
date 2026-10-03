# Python files of the lecture notes

    figures/ch1/   scripts producing the figures of Chapter 1 (output: ../../../figures/ch1/)
    figures/appB/  script producing the Gateaux/Frechet figure of Appendix B
    examples/ch1/  computational exercises of Chapter 1 and the constrained spring chain
    examples/ch2/  computational exercises of Chapter 2

The exercise files are included verbatim in the book with `\lstinputlisting`,
so editing a file here changes the printed solution. Each file runs on its own:

    pip install numpy scipy sympy matplotlib
    python3 python/examples/ch1/exo_gd.py
    cd python/figures/ch1 && python3 fig_resonance.py
