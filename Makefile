JQ = jq --indent 4 -r
PYENV_ROOT = $(HOME)/.pyenv
POETRY = $(HOME)/.poetry/bin/poetry

DOCKER_IMAGE_TAG = bmed8813rob-sp21-team1/riddbot:0.0.0
DOCKER_VOLUMES = -v `bin`:`/workspace/bin` -v `configs`:`/workspace/configs` -v `riddbot`:`/workspace/riddbot`
DOCKER_PORTS = -p 8265:8265
DOCKER_MAKE = docker run $(DOCKER_VOLUMES) $(DOCKER_PORTS) $(DOCKER_IMAGE_TAG)


.PHONY: clean_python
clean_python:
	rm -rfv **/__pycache__
	rm -rfv **/.ipynb_checkpoints

.PHONY: clean
clean: clean_python clean_poetry

.PHONY: docker_image
docker_image:
	docker build . -t $(DOCKER_IMAGE_TAG)

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
	$(POETRY) lock -vvv

.venv: poetry.lock | $(POETRY)
	$(POETRY) run pip install pip==21.0.1
	$(POETRY) install -vvv

.PHONY: python_deps
python_deps: .venv

.PHONY: clean_poetry
clean_poetry:
	rm -rf .venv
	rm -f poetry.lock

.PHONY: jupyterlab
jupyterlab: | .venv
	$(POETRY) run jupyter lab

.PHONY: jupyterlab_server
jupyterlab_server: | .venv
	$(POETRY) run jupyter lab --no-browser --ip="0.0.0.0"

.PHONY: configs/%.json.train
configs/%.json.train: configs/%.json | .venv
	$(POETRY) run python bin/train_model.py $<

.PHONY: configs/%.json.train_docker
configs/%.json.train_docker: configs/%.json
	$(DOCKER_MAKE) $<.train
