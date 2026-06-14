# LATEX: pdflatex | xelatex | lualatex
LATEX = lualatex
LATEX_FLAGS = -shell-escape
BIBTEX = bibtex

FILES = $(patsubst %.tex, %.pdf, $(wildcard *.tex))

all: $(FILES)


%.pdf: %.tex
	$(LATEX) $(LATEX_FLAGS) $<
	$(LATEX) $(LATEX_FLAGS) $<
	-$(BIBTEX) `basename $< .tex`
	$(LATEX) $(LATEX_FLAGS) $<
	$(LATEX) $(LATEX_FLAGS) $<
	$(LATEX) $(LATEX_FLAGS) $<
	@if command -v qpdf >/dev/null 2>&1; then \
		qpdf --linearize --newline-before-endstream $@ /tmp/$@ && mv /tmp/$@ $@; \
	else \
		echo "qpdf not found; skipping PDF linearization (PDF is still built)"; \
	fi

clean:
	-rm -f *.{log,toc,tac,aux,dvi,ps,bbl,blg,tmp,nav,out,snm,vrb,rel,cut,abs,xmpi,xmpdata,*~}
	-rm -rf out _minted* auto
	for i in $(ALLSUBDIRS); do \
	    (cd $$i; make clean) || exit 1; \
	done
	-rm $(FILES)

cleanall: clean
	-rm -rf out auto