.PHONY: setup clean ingest

setup:
	uv sync

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf vector_db/
	@echo "Cleanup complete."

ingest:
	uv run python -m src.prepare_vector_db.ingest


format:
	$(PYTHON) -m black src/