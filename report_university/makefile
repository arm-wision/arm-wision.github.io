# author: Pascal Bercher, pascal.bercher@anu.edu.au, (28.12.2025)

STATE = .buildtarget
LONG  = main-long
SHORT = main-short

TARGET := $(shell if [ -f $(STATE) ]; then cat $(STATE); else echo $(LONG); fi)

TEXFILE = $(TARGET).tex
BIBFILE = $(TARGET).aux

LATEX   = pdflatex
BIBTEX  = bibtex

LATEXMK = latexmk

TRASH = *.aux *.log *.dvi *.ps *.nav *.out *.snm *.backup *.bak *.toc *~ *.bbl *.blg *.toc *.fls *.brf *.fdb_latexmk *.synctex.gz

.PHONY: default
default: mkonline

init:
	@if [ ! -f $(STATE) ]; then \
		echo "$(LONG)" > $(STATE); \
		echo "No build target set — defaulting to: $(LONG).tex"; \
	fi

long:
	@echo "$(LONG)" > $(STATE)
	@echo "Build target set to: $(LONG).tex"

short:
	@echo "$(SHORT)" > $(STATE)
	@echo "Build target set to: $(SHORT).tex"

which:
	@echo "Current build target: $(TARGET).tex"

mkonline: init
	$(LATEXMK) -pvc -pdf $(TEXFILE)

mk: init
	$(LATEXMK) -pdf $(TEXFILE)

all: init
	$(LATEX) -halt-on-error $(TEXFILE)
	$(BIBTEX) $(BIBFILE)
	$(LATEX) -halt-on-error $(TEXFILE)
	$(LATEX) -halt-on-error $(TEXFILE)
	$(LATEX) -halt-on-error $(TEXFILE)

quick: init
	$(LATEX) -halt-on-error $(TEXFILE)

clean: clear

clear:
	rm -rf $(TRASH)