PYTHON ?= python3
PIP ?= $(PYTHON) -m pip
BLACK ?= black
RUFF ?= ruff
PYTEST ?= pytest

APP_DIRS = app tests

.PHONY: install install-dev format lint check test docker-build docker-up docker-down docker-logs full-test

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-enterprise.txt

install-dev: install
	$(PIP) install -r requirements-dev.txt

format:
	$(BLACK) $(APP_DIRS)

lint:
	$(RUFF) check $(APP_DIRS)

check:
	$(BLACK) --check $(APP_DIRS)
	$(RUFF) check $(APP_DIRS)

test:
	$(PYTEST) tests

pytest-contracts:
	$(PYTEST) tests/test_realtime_routes.py

docker-build:
	docker compose -f docker-compose.yml build

docker-up:
	docker compose -f docker-compose.yml up -d

docker-down:
	docker compose -f docker-compose.yml down -v

docker-logs:
	docker compose -f docker-compose.yml logs -f --tail=200

full-test: docker-build docker-up
	@echo "Waiting for services to become healthy..."
	@sleep 15
	$(PYTEST) tests
	$(MAKE) docker-down
