
# Handling conda environment in Makefile:
# 	https://stackoverflow.com/questions/53382383/makefile-cant-use-conda-activate
# 	Makefile can't use `conda activate`

.ONESHELL:

SHELL = /bin/bash

CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

step01:
	$(CONDA_ACTIVATE) jupyter_data_processing02
	python src/s01_generate_data.py
	conda deactivate

step02:
	$(CONDA_ACTIVATE) jupyter_data_processing02
	python src/s02_model_train.py
	conda deactivate

step03:
	$(CONDA_ACTIVATE) jupyter_data_processing02
	python src/s03_model_eval.py
	conda deactivate

step03_all: step03 step02
