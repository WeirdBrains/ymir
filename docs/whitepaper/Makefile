# Build the Ymir white paper.  Needs a TeX distribution with latexmk, pgfplots and TikZ.
MAIN := ymir-whitepaper

.PHONY: all pdf manual clean distclean
all: pdf

pdf:
	latexmk -pdf $(MAIN).tex

# Without latexmk: pdflatex, bibtex, then pdflatex twice.
manual:
	pdflatex $(MAIN) && bibtex $(MAIN) && pdflatex $(MAIN) && pdflatex $(MAIN)

clean:
	latexmk -c $(MAIN).tex

distclean:
	latexmk -C $(MAIN).tex
