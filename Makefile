.PHONY: lint test

lint:
	pre-commit run --all-files

test:
	uv run pytest
