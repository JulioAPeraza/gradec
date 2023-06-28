.PHONY: all lint

all_tests: lint unittest performancetest

help:
	@echo "Please use 'make <target>' where <target> is one of:"
	@echo "  lint							to run flake8 on all Python files"
	@echo "  unittest						to run unit tests on nimare"

lint:
	@flake8 gradec

unittest:
	@py.test --cov-append --cov-report=xml --cov=gradec gradec
