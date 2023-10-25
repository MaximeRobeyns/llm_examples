.PHONY: finetune install board lab kernel help

finetune:  ## Example model fine-tuning script on symbol binding task
	accelerate launch --config_file configs/accelerate/local_with_ds.yaml \
		llm_examples/mcsb_finetune.py

install:  ## Install the project locally
	pip install -e .

board:  ## Run tensorboard
	tensorboard --logdir=logs

lab:  ## To start a Jupyter Lab server
	jupyter lab --notebook-dir=notebooks --ip=0.0.0.0 --port 9001

kernel:  ## To setup a Jupyter kernel to run notebooks in the project's virtual env
	python -m ipykernel install --user --name llm_venv \
		--display-name "llm_venv (Python 3.10)"

help:
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
