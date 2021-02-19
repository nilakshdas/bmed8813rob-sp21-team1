SUPPORTED_PYTHON_VERSION ?= 3.6
CURRENT_PYTHON_VERSION = $(shell python -c "import sys; print('%d.%d' % sys.version_info[0:2]);")

ifneq ($(CURRENT_PYTHON_VERSION),$(SUPPORTED_PYTHON_VERSION))
  $(error "Supported python version is $(REQUIRED_PYTHON_VERSION), current version is $(CURRENT_PYTHON_VERSION)")
endif

POETRY = $(HOME)/.poetry/bin/poetry


.PHONY: clean
clean:
	rm -rf .venv

$(POETRY):
	curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

.PHONY: poetry
poetry: $(POETRY)
	$(POETRY) $(ARGS)

poetry.lock: $(POETRY) pyproject.toml
	$(POETRY) lock

.venv: $(POETRY) poetry.lock
	$(POETRY) install
	touch $@

.PHONY: relock_poetry
relock_poetry:
	rm -f poetry.lock
	$(MAKE) poetry.lock

.PHONY: python_deps
python_deps: .venv
