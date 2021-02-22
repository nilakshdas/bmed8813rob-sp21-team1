CURRENT_PYTHON_VERSION = $(shell python -V)
SUPPORTED_PYTHON_VERSION = $(shell echo "Python" | cat - .python-version)

ifneq ($(CURRENT_PYTHON_VERSION),$(SUPPORTED_PYTHON_VERSION))
  $(error "Supported python version is $(SUPPORTED_PYTHON_VERSION), current version is $(CURRENT_PYTHON_VERSION)")
endif


JQ = jq --indent 4 -r
PYENV_ROOT = $(HOME)/.pyenv
POETRY = $(HOME)/.poetry/bin/poetry


.PHONY: clean_python
clean_python:
	rm -rfv **/__pycache__
	rm -rfv **/.ipynb_checkpoints

.PHONY: clean
clean: clean_python clean_poetry

$(PYENV_ROOT):
	curl -sSL https://raw.githubusercontent.com/pyenv/pyenv-installer/master/bin/pyenv-installer | bash

.PHONY: pyenv
pyenv: | $(PYENV_ROOT)
	pyenv install

$(POETRY):
	curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

.PHONY: poetry
poetry: $(POETRY)
	$(POETRY) $(ARGS)

poetry.lock: pyproject.toml | $(POETRY)
	$(POETRY) lock

.venv: poetry.lock | $(POETRY)
	$(POETRY) install

.PHONY: python_deps
python_deps: .venv

.PHONY: clean_poetry
clean_poetry:
	rm -rf .venv
	rm -f poetry.lock

.PHONY: lint_check
lint_check: | .venv
	$(POETRY) run black --check .

.PHONY: lint
lint: | .venv
	$(POETRY) run black .

.PHONY: jupyterlab
jupyterlab: | .venv
	$(POETRY) run jupyter lab

.PHONY: configs/%.json.train
configs/%.json.train: configs/%.json
	$(POETRY) run python bin/train_model.py $<
