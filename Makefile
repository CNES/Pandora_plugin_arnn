# Autodocumented Makefile
# see: https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
#
# Dependencies : python3 venv internal module
# Recall: .PHONY  defines special targets not associated with files
#
# Some Makefile global variables can be set in make command line:
# PLUGIN_ARNN_VENV: Change directory of installed venv (default local "venv" dir)
#

############### GLOBAL VARIABLES ######################

.DEFAULT_GOAL := help
# Set shell to BASH
SHELL := /bin/bash

# Set Virtualenv directory name
# Example: PLUGIN_ARNN_VENV="other-venv/" make install
ifndef PLUGIN_ARNN_VENV
	PLUGIN_ARNN_VENV = "venv"
endif

# Check if plugin is installed
CHECK_PLUGIN_ARNN = $(shell ${PLUGIN_ARNN_VENV}/bin/python3 -m pip list|grep pandora_plugin_arnn)

# Check python3 globally
PYTHON=$(shell command -v python3)
ifeq (, $(PYTHON))
    $(error "PYTHON=$(PYTHON) not found in $(PATH)")
endif

# Check Python version supported globally
PYTHON_VERSION_MIN = 3.8
PYTHON_VERSION_CUR=$(shell $(PYTHON) -c 'import sys; print("%d.%d"% sys.version_info[0:2])')
PYTHON_VERSION_OK=$(shell $(PYTHON) -c 'import sys; cur_ver = sys.version_info[0:2]; min_ver = tuple(map(int, "$(PYTHON_VERSION_MIN)".split("."))); print(int(cur_ver >= min_ver))')
ifeq ($(PYTHON_VERSION_OK), 0)
    $(error "Requires python version >= $(PYTHON_VERSION_MIN). Current version is $(PYTHON_VERSION_CUR)")
endif


################ MAKE targets by sections ######################

.PHONY: help
help: ## this help
	@echo "      PANDORA MAKE HELP"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'| sort

## Install section

.PHONY: venv
venv: ## create virtualenv in PLUGIN_ARNN_VENV directory if not exists
	@test -d ${PLUGIN_ARNN_VENV} || python3 -m venv ${PLUGIN_ARNN_VENV}
	@${PLUGIN_ARNN_VENV}/bin/python -m pip install --upgrade pip setuptools wheel # no check to upgrade each time
	@touch ${PLUGIN_ARNN_VENV}/bin/activate

.PHONY: install
install: venv ## install pandora_plugin_arnn (pip editable mode) without plugins
	@[ "${CHECK_PLUGIN_ARNN}" ] || ${PLUGIN_ARNN_VENV}/bin/pip install -e .[dev,docs]
	@test -f .git/hooks/pre-commit || echo "  Install pre-commit hook"
	@test -f .git/hooks/pre-commit || ${PLUGIN_ARNN_VENV}/bin/pre-commit install
	@echo "PANDORA installed in dev mode in virtualenv ${PLUGIN_ARNN_VENV}"
	@echo "PANDORA venv usage : source ${PLUGIN_ARNN_VENV}/bin/activate; python -c 'import pandora_plugin_arnn' "

## Test section

.PHONY: test
test: install ## run all tests + coverage (source venv before)
	@${PLUGIN_ARNN_VENV}/bin/pytest --junitxml=pytest-report.xml --cov-config=.coveragerc --cov-report xml --cov

## Code quality, linting section

### Format with black

.PHONY: format
format: install format/black  ## run black formatting (depends install)

.PHONY: format/black
format/black: install  ## run black formatting (depends install) (source venv before)
	@echo "+ $@"
	@${PLUGIN_ARNN_VENV}/bin/black pandora_plugin_arnn tests ./*.py

### Check code quality and linting : black, mypy, pylint

.PHONY: lint
lint: install lint/black lint/mypy lint/pylint ## check code quality and linting (source venv before)

.PHONY: lint/black
lint/black: ## check global style with black
	@echo "+ $@"
	@${PLUGIN_ARNN_VENV}/bin/black --check pandora_plugin_arnn tests ./*.py

.PHONY: lint/mypy
lint/mypy: ## check linting with mypy
	@echo "+ $@"
	@${PLUGIN_ARNN_VENV}/bin/mypy pandora_plugin_arnn tests

.PHONY: lint/pylint
lint/pylint: ## check linting with pylint
	@echo "+ $@"
	@set -o pipefail; ${PLUGIN_ARNN_VENV}/bin/pylint pandora_plugin_arnn tests --rcfile=.pylintrc --output-format=parseable --msg-template="{path}:{line}: [{msg_id}({symbol}), {obj}] {msg}" | tee pylint-report.txt # pipefail to propagate pylint exit code in bash

## Clean section

.PHONY: clean
clean: clean-venv clean-build clean-precommit clean-pyc clean-test clean-mypy ## remove all build, test, coverage and Python artifacts

.PHONY: clean-venv
clean-venv:
	@echo "+ $@"
	@rm -rf ${PLUGIN_ARNN_VENV}

.PHONY: clean-build
clean-build:
	@echo "+ $@"
	@rm -fr build/
	@rm -fr .eggs/
	@find . -name '*.egg-info' -exec rm -fr {} +
	@find . -name '*.egg' -exec rm -f {} +

.PHONY: clean-precommit
clean-precommit:
	@rm -f .git/hooks/pre-commit
	@rm -f .git/hooks/pre-push

.PHONY: clean-pyc
clean-pyc:
	@echo "+ $@"
	@find . -type f -name "*.py[co]" -exec rm -fr {} +
	@find . -type d -name "__pycache__" -exec rm -fr {} +
	@find . -name '*~' -exec rm -fr {} +

.PHONY: clean-test
clean-test:
	@echo "+ $@"
	@rm -fr .tox/
	@rm -f .coverage
	@rm -rf .coverage.*
	@rm -rf coverage.xml
	@rm -fr htmlcov/
	@rm -fr .pytest_cache
	@rm -f pytest-report.xml
	@rm -f pylint-report.txt
	@rm -f debug.log


.PHONY: clean-mypy
clean-mypy:
	@echo "+ $@"
	@rm -rf .mypy_cache/
