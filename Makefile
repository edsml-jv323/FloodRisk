
format:
	ruff --fix-only .
	black .

lint:
	ruff check .
	black --check --diff .
