.PHONY: all artifacts pdf clean
BUILD := build
PYTHON ?= ../.venv/bin/python
all: artifacts pdf
artifacts:
	$(PYTHON) scripts/build_artifacts.py
	$(PYTHON) scripts/build_results_figures.py
pdf:
	mkdir -p $(BUILD)
	pdflatex -synctex=1 -file-line-error -interaction=nonstopmode -halt-on-error -output-directory=$(BUILD) main.tex > $(BUILD)/build-pass1.log
	bibtex $(BUILD)/main > $(BUILD)/build-bibtex.log
	pdflatex -synctex=1 -file-line-error -interaction=nonstopmode -halt-on-error -output-directory=$(BUILD) main.tex > $(BUILD)/build-pass2.log
	pdflatex -synctex=1 -file-line-error -interaction=nonstopmode -halt-on-error -output-directory=$(BUILD) main.tex > $(BUILD)/build-pass3.log
	if grep -q 'Rerun to get' $(BUILD)/main.log; then pdflatex -synctex=1 -file-line-error -interaction=nonstopmode -halt-on-error -output-directory=$(BUILD) main.tex > $(BUILD)/build-pass4.log; fi
clean:
	rm -rf $(BUILD)
