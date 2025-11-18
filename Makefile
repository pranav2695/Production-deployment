export_env:
	conda env export --from-history --name workshop-dev > deploy/conda/env.yml

create_env:
	conda env create -f ./deploy/conda/env.yml

format-check: setup.cfg
	black --check .
	isort --check-only --diff .
	flake8 .

format: setup.cfg
	isort .
	black .
	flake8 .

test:
	pytest -v

clean:
	find . -type d -name '__pycache__' -exec rm -r {} +
	find . -type d -name '.pytest_cache' -exec rm -r {} +

serve_docs:
	python3 -m http.server 4005 -d docs/build/html

run:
	python3 scripts/main.py

clean_pkg:
	rm -rf ./dist ./src/tamlep.egg-info

build_pkg:
	python3 -m build

install_pkg:
	pip3 install -e .

docker_build:
	docker build --no-cache -t tamlep:latest -f ./deploy/docker/Dockerfile .
