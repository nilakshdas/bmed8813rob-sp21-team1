JQ = jq --indent 4 -r
PYENV_ROOT = $(HOME)/.pyenv
PYENV = $(HOME)/.pyenv/bin/pyenv
POETRY = $(HOME)/.poetry/bin/poetry


.PHONY: clean_python
clean_python:
	rm -rfv **/__pycache__
	rm -rfv **/.ipynb_checkpoints

.PHONY: clean
clean: clean_python clean_poetry

$(PYENV_ROOT):
	curl -sSL https://raw.githubusercontent.com/pyenv/pyenv-installer/master/bin/pyenv-installer | bash
	@echo && echo "Press any key to continue..." && read key

.PHONY: pyenv
pyenv: | $(PYENV_ROOT)
	$(PYENV) install

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
	touch $@ # update timestamp

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

configs/%_t5M.json: configs/%.json
	cat $< | $(JQ) '.train_timesteps = 5000000' > $@

.PHONY: configs/%.json.train
configs/%.json.train: configs/%.json | .venv
	$(POETRY) run python bin/train_model.py $<

.PHONY: configs/%.json.render
configs/%.json.render: configs/%.json | .venv
	$(POETRY) run python bin/render.py --config_path $< --output_dir out/

.PHONY: riddbot/model_zoo/%/.render
riddbot/model_zoo/%/.render: riddbot/model_zoo/%
	$(POETRY) run python bin/render.py --model_path $<
