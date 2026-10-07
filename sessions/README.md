# Exercise sessions

One PDF per session, in the order of use:

1. the session sheet: questions only (at most 8 pages, printed); off for
   now, `\questionsheettrue` in `session_ch3.tex` puts it back;
2. the session sheet with the answers;
3. the guides of the three codes, with the same blocks and the same
   structure (running, parameters, blocks, figures, printed values,
   pitfalls, listing): Cast3M, FEniCS, Python.

The sheet is written once (`ch3/sheet.tex`) and printed twice: the
`answer` environments appear only in the second copy (`\answerstrue`).

    session_preamble.tex   header "Nonlinear Methods in Mechanics - 2026",
                           parts a, b, ... and questions a.1, ..., answers
    session_ch3.tex        Chapter 3, reciprocity gap
    ch3/sheet.tex          questions and answers (parts a-g)
    ch3/codes.tex          common to the three guides: method, blocks,
                           figures, reference values
    ch3/guide_*.tex        Cast3M, FEniCS, Python guides
    ch3/figs/python/       figures drawn by python/examples/ch3/rg_session.py

Codes: `cast3m/ch3/rg_session.dgibi`, `fenics/ch3/rg_session_fenics.py`,
`python/examples/ch3/rg_session.py`. Blocks 4-9 of the FEniCS and Python
files are the same text: edit the Python file and copy them across.

## Build

The book must be compiled first (its `.aux` files give the numbers cited,
prefix `B-`). Then

    cd sessions && latexmk -pdf session_ch3

The CI builds it after the book and attaches `NPM_session_ch3.pdf` to the
release "latest".
