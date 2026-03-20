.PHONY: setup clean ingest answer evaluate evaluation_app

TEST_ROW=2

setup:
	uv sync

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf vector-db/
	@echo "Cleanup complete."

ingest:
	uv run python -m src.prepare_vector_db.ingest

answer:
	uv run python -m src.answering.answer


evaluate:
	uv run python -m src.evaluation.evaluate ${TEST_ROW}

evaluation_app:
	uv run python -m src.evaluation.evaluator_ui


format:
	$(PYTHON) -m black src/