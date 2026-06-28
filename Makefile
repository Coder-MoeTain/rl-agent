.PHONY: install validate smoke scan live train docker-scan clean

install:
	pip install -e ".[dev]"

validate:
	autopentest validate --offline

smoke:
	autopentest scan --mock --algorithm rule_based -o ./reports/smoke -q

scan:
	autopentest scan --authorized -o auto --algorithm auto

live:
	docker compose up -d juice-shop
	autopentest health -t http://localhost:3000
	autopentest scan --authorized -t http://localhost:3000 -o auto

train:
	autopentest train --algorithm ppo_per

docker-scan:
	docker compose build pentester
	docker compose --profile scan up

test:
	pytest tests/test_production_pentester.py -v

clean:
	rm -rf reports/smoke reports/ci_smoke reports/test_production reports/cli_test
