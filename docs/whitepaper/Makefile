# Build the Ymir white paper.  Needs TeX Live with TikZ and pgfplots.
# Uses latexmk when it is installed, otherwise pdflatex + bibtex.
MAIN := ymir-whitepaper
LATEXMK := $(shell command -v latexmk 2>/dev/null)
PDFLATEX := pdflatex -interaction=nonstopmode -halt-on-error

.PHONY: all pdf manual clean distclean
all: pdf

pdf:
ifneq ($(LATEXMK),)
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(MAIN).tex
else
	$(MAKE) manual
endif

# Without latexmk: pdflatex, bibtex, then pdflatex twice.
manual:
	$(PDFLATEX) $(MAIN) && bibtex $(MAIN) && $(PDFLATEX) $(MAIN) && $(PDFLATEX) $(MAIN)

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).fdb_latexmk $(MAIN).fls $(MAIN).log $(MAIN).out $(MAIN).toc $(MAIN).lof $(MAIN).lot $(MAIN).synctex.gz

distclean: clean
	rm -f $(MAIN).pdf
