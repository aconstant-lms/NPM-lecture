# Build the Cast3M companion. The book (../main.tex) must have been compiled
# first: its .aux files give the numbers of the chapters, exercises and boxes
# cited here (package xr-hyper).
all:
	latexmk -pdf cast3m_companion
book:
	cd .. && pdflatex -interaction=nonstopmode main >/dev/null && pdflatex -interaction=nonstopmode main >/dev/null
check:
	@grep -c "undefined" cast3m_companion.log || true
	@grep -c "Overfull" cast3m_companion.log || true
clean:
	latexmk -C cast3m_companion
