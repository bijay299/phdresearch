.PHONY: test smoke pilot

# Regression tests. Run after touching src/metrics.py or src/data.py.
# Fails (nonzero exit, make stops) if either test file fails.
test:
	python src/test_metrics.py
	python src/test_data.py

# Wiring check, CPU, ~2 min.
smoke:
	python scripts/run_experiment.py --config configs/cifar_ce.yaml --smoke

# The pilot pair: same everything except the head.
pilot:
	python scripts/run_experiment.py --config configs/cifar_ce.yaml
	python scripts/run_experiment.py --config configs/cifar_arcface.yaml
