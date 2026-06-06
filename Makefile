.PHONY: test benchmark

test:
	python -m unittest discover -s tests

benchmark:
	python scripts/run_benchmark.py
