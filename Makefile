.PHONY: setup clean ingest chatbot answer evaluate evaluation_app

TEST_ROW=2

setup:
	uv sync

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf vector-db/
	@echo "Cleanup complete."

# ingest the knowledge base to vector data base
ingest:
	uv run python -m src.prepare_vector_db.ingest

# launch the chatbot
chatbot:
	uv run python -m chatbot

# answer a question
answer:
	uv run python -m src.answering.answer

# Test on specific question by assigning the test row
evaluate:
	uv run python -m src.evaluation.evaluate ${TEST_ROW}

# Go through all test set, generate metrics plots
evaluation_app:
	uv run python -m src.evaluation.evaluator_ui

format:
	black src