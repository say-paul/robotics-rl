.PHONY: help install dry-run-sonic dry-run-decoupled download-model download-sonic \
       isaac-build isaac-run isaac-push isaac-deploy isaac-undeploy clean

GROOT_WBC ?= $(HOME)/redhat/git/GR00T-WholeBodyControl
REGISTRY ?= quay.io/jary
NAMESPACE ?= robotics-rl

help:
	@echo "make install             Install Python dependencies"
	@echo "make dry-run-sonic       Dry-run SONIC WBC config"
	@echo "make dry-run-decoupled   Dry-run decoupled WBC config"
	@echo "make download-model      Download GR00T N1.6 VLA model"
	@echo "make download-sonic      Download GEAR-SONIC WBC models"
	@echo "make isaac-build         Build Isaac Sim container"
	@echo "make isaac-run           Run Isaac Sim container locally"
	@echo "make isaac-push          Push images to registry"
	@echo "make isaac-deploy        Deploy to OpenShift"
	@echo "make isaac-undeploy      Remove from OpenShift"
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

isaac-build:
	docker build -f isaac_sim/Containerfile -t rdp-isaac-sim isaac_sim/
	docker build -f isaac_sim/Containerfile.viewer -t rdp-isaac-viewer isaac_sim/

isaac-run:
	docker run -it --rm --gpus all --network=host --cap-add=NET_ADMIN -v $(GROOT_WBC):/groot -v $(CURDIR):/rdp rdp-isaac-sim

isaac-push:
	docker tag rdp-isaac-sim $(REGISTRY)/rdp-isaac-sim:latest
	docker tag rdp-isaac-viewer $(REGISTRY)/rdp-isaac-viewer:latest
	docker push $(REGISTRY)/rdp-isaac-sim:latest
	docker push $(REGISTRY)/rdp-isaac-viewer:latest

isaac-deploy:
	oc apply -f isaac_sim/k8s/namespace.yaml
	oc apply -f isaac_sim/k8s/deployment.yaml
	oc apply -f isaac_sim/k8s/service.yaml
	oc apply -f isaac_sim/k8s/route.yaml
	@echo "Viewer URL:"
	@oc get route g1-viewer -n $(NAMESPACE) -o jsonpath='{.spec.host}' 2>/dev/null && echo || echo "(pending)"

isaac-undeploy:
	oc delete -f isaac_sim/k8s/route.yaml --ignore-not-found
	oc delete -f isaac_sim/k8s/service.yaml --ignore-not-found
	oc delete -f isaac_sim/k8s/deployment.yaml --ignore-not-found

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info
