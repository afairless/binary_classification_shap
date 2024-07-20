
# Handling conda environment in Makefile:
# 	https://stackoverflow.com/questions/53382383/makefile-cant-use-conda-activate
# 	Makefile can't use `conda activate`

.ONESHELL:

SHELL = /bin/bash

CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

step01:
	$(CONDA_ACTIVATE) jupyter_data_processing02
	python src/s01_clean_data.py
	conda deactivate

step02:
	$(CONDA_ACTIVATE) jupyter_data_processing02
	python src/s02_recode_data.py
	conda deactivate

step02_all: step01 step02

step03:
	$(CONDA_ACTIVATE) jupyter_data_processing02
	python src/s03_eda.py
	conda deactivate

step03_all: step02 step03

step04:
	$(CONDA_ACTIVATE) jupyter_data_processing02
	python src/s04_model_train.py
	conda deactivate

step04_all: step03 step04

step05:
	$(CONDA_ACTIVATE) jupyter_data_processing02
	python src/s05_model_eval.py
	conda deactivate

step05_all: step04 step05

notebook:
	$(CONDA_ACTIVATE) jupyter_data_processing02
	cp -r output notebooks/output
	jupytext --to notebook notebooks/*.py
	conda deactivate

notebook_all: step05 notebooks

