# Makefile for HFT Trading System

.PHONY: build up down logs train backtest lint test

build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down

logs:
	docker-compose logs -f

train:
	docker exec -it hft_app python model_runner.py

backtest:
	docker exec -it hft_app python backtest.py

lint:
	black . && flake8 .

test:
	pytest tests/