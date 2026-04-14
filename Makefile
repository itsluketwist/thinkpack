# simple make file for common commands

new: newvenv newlint

check: lint test

newvenv:
	python -m venv .venv
	. venv/bin/activate

newlint:
	pre-commit install
	pre-commit autoupdate

lint:
	pre-commit run --all-files

test:
	pytest tests

coverage:
	pytest --cov=src/ tests/

bundle:
	cp llms.txt src/thinkpack/data/llms.txt
