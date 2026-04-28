.PHONY: help install dry-run-sonic dry-run-decoupled download-model download-sonic clean

help:
	@echo "make install             Install Python dependencies"
	@echo "make dry-run-sonic       Dry-run SONIC WBC config"
	@echo "make dry-run-decoupled   Dry-run decoupled WBC config"
	@echo "make download-model      Download GR00T N1.6 VLA model"
	@echo "make download-sonic      Download GEAR-SONIC WBC models"
	@echo "make clean               Remove __pycache__ and build artifacts"

install:
	pip install -r requirements.txt

dry-run-sonic:
	python scripts/launch_robot.py --robot configs/robots/g1_sonic_wbc.yaml --dry-run

dry-run-decoupled:
	python scripts/launch_robot.py --robot configs/robots/g1_groot_wbc_unified.yaml --dry-run

download-model:
	python scripts/download_groot_model.py --version n1.6 --verify

download-sonic:
	pip install -q huggingface-hub
	python scripts/download_groot_model.py --sonic --verify

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info
