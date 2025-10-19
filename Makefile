.PHONY: lint test dashboard

lint:
	pre-commit run --all-files

test:
	uv run pytest

dashboard:
	uv run streamlit run src/exploration_sequential_testing/visualization/dashboard/app.py
