import sys
import math
from pathlib import Path
from typing import Tuple
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

from src.evaluation.test import TestQuestion, load_tests
from src.answering.answer import answer_question, retrieve_chunks
from src.utils.pydantic_models import Answer, RetrievalEval, AnswerEval
from src.utils.prompts import evaluation_prompt

load_dotenv(override=True)

DEBUG = True

KNOWLEDGE_BASE_DIR = str(Path(__file__).parent.parent.parent / "knowledge-base")
VECTOR_DB_NAME = str(Path(__file__).parent.parent.parent / "vector-db")
COLLECTION_NAME = "docs"

# Embedding model for VectorDB
EMBEDDING_METHOD = "huggingface"  # or "openai"
if EMBEDDING_METHOD == "huggingface":
    EMBEDDING = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
elif EMBEDDING_METHOD == "openai":
    EMBEDDING = OpenAIEmbeddings(
        model="text-embedding-3-small",
        model_kwargs={
            "temperature": 0.0,
            "max_tokens": 2048,
            "top_p": 1.0,
        },
    )

VECTOR_DB = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=VECTOR_DB_NAME,
    embedding_function=EMBEDDING,
)

# Answering LLM
QUESTION_ANSERING_LLM = "gpt-4.1-mini"
LLM = ChatOpenAI(
    model=QUESTION_ANSERING_LLM, temperature=0.0, max_tokens=2048, top_p=1.0
).with_structured_output(Answer)

# Evaluation LLM
EVALUATION_LLM = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
).with_structured_output(AnswerEval)


# class RetrievalEval(BaseModel):
#     """Evaluation metrics for retrieval performance."""

#     mrr: float = Field(description="Mean Reciprocal Rank - average across all keywords")
#     ndcg: float = Field(description="Normalized Discounted Cumulative Gain (binary relevance)")
#     keywords_found: int = Field(description="Number of keywords found in top-k results")
#     total_keywords: int = Field(description="Total number of keywords to find")
#     keyword_coverage: float = Field(description="Percentage of keywords found")


# class AnswerEval(BaseModel):
#     """LLM-as-a-judge evaluation of answer quality."""

#     feedback: str = Field(
#         description="Concise feedback on the answer quality, comparing it to the reference answer and evaluating based on the retrieved context"
#     )
#     accuracy: float = Field(
#         description="How factually correct is the answer compared to the reference answer? 1 (wrong. any wrong answer must score 1) to 5 (ideal - perfectly accurate). An acceptable answer would score 3."
#     )
#     completeness: float = Field(
#         description="How complete is the answer in addressing all aspects of the question? 1 (very poor - missing key information) to 5 (ideal - all the information from the reference answer is provided completely). Only answer 5 if ALL information from the reference answer is included."
#     )
#     relevance: float = Field(
#         description="How relevant is the answer to the specific question asked? 1 (very poor - off-topic) to 5 (ideal - directly addresses question and gives no additional information). Only answer 5 if the answer is completely relevant to the question and gives no additional information."
#     )


def calculate_mrr(keyword: str, retrieved_chunks: list) -> float:
    """Calculate reciprocal rank for a single keyword (case-insensitive)."""
    keyword_lower = keyword.lower()
    for rank, doc in enumerate(retrieved_chunks, start=1):
        if keyword_lower in doc.page_content.lower():
            return 1.0 / rank
    return 0.0


def calculate_dcg(relevances: list[int], top_k: int) -> float:
    """Calculate Discounted Cumulative Gain."""
    dcg = 0.0
    for i in range(min(top_k, len(relevances))):
        dcg += relevances[i] / math.log2(i + 2)  # i+2 because rank starts at 1
    return dcg


def calculate_ndcg(keyword: str, retrieved_docs: list, top_k: int = 5) -> float:
    """Calculate nDCG for a single keyword (binary relevance, case-insensitive)."""
    keyword_lower = keyword.lower()

    # Binary relevance: 1 if keyword found, 0 otherwise
    relevances = [
        1 if keyword_lower in doc.page_content.lower() else 0
        for doc in retrieved_docs[:top_k]
    ]

    # DCG
    dcg = calculate_dcg(relevances, top_k)

    # Ideal DCG (best case: keyword in first position)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = calculate_dcg(ideal_relevances, top_k)

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(test: TestQuestion, top_k: int = 5) -> RetrievalEval:
    """
    Evaluate retrieval performance for a test question.

    Args:
        test: TestQuestion object containing question and keywords
        top_k: Number of top documents to retrieve (default 10)

    Returns:
        RetrievalEval object with MRR, nDCG, and keyword coverage metrics
    """
    # Retrieve documents using shared answer module
    retrieved_docs = retrieve_chunks(
        test.question, vector_db=VECTOR_DB, top_k=top_k, embedding=EMBEDDING
    )

    # Calculate MRR (average across all keywords)
    mrr_scores = [calculate_mrr(keyword, retrieved_docs) for keyword in test.keywords]
    avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0

    # Calculate nDCG (average across all keywords)
    ndcg_scores = [
        calculate_ndcg(keyword, retrieved_docs, top_k) for keyword in test.keywords
    ]
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0

    # Calculate keyword coverage
    keywords_found = sum(1 for score in mrr_scores if score > 0)
    total_keywords = len(test.keywords)
    keyword_coverage = (
        (keywords_found / total_keywords * 100) if total_keywords > 0 else 0.0
    )

    return RetrievalEval(
        mrr=avg_mrr,
        ndcg=avg_ndcg,
        keywords_found=keywords_found,
        total_keywords=total_keywords,
        keyword_coverage=keyword_coverage,
    )


def evaluate_answer(test: TestQuestion, top_k: int = 5):
    """
    Evaluate answer quality using LLM-as-a-judge (async).

    Args:
        test: TestQuestion object containing question and reference answer


    Returns:
        Tuple of (AnswerEval object, generated_answer string, reference answer string)
    """
    # Get RAG response using shared answer module
    generated_answer, _ = answer_question(
        test.question, history=[], top_k=top_k, llm_model=LLM
    )

    sys_prompt, user_prompt = evaluation_prompt(
        question=test.question,
        reference_answer=test.reference_answer,
        answer=generated_answer,
    )

    eval_messages = [
        SystemMessage(content=sys_prompt),
        HumanMessage(content=user_prompt),
    ]
    # Call LLM judge with structured outputs (async)
    eval_response = EVALUATION_LLM.invoke(
        input=eval_messages  # here for google model, the parameter is 'input' not 'messages'
    )

    return eval_response, generated_answer, test.reference_answer


def evaluate_all_retrieval():
    """Evaluate all retrieval tests."""
    tests = load_tests()[0:100]
    total_tests = len(tests)
    for index, test in enumerate(tests):
        result = evaluate_retrieval(test=test)
        progress = (index + 1) / total_tests
        yield test, result, progress


def evaluate_all_answers():
    """Evaluate all answers to tests using batched async execution."""
    tests = load_tests()[0:100]
    total_tests = len(tests)
    for index, test in enumerate(tests):
        result = evaluate_answer(test)[0]
        progress = (index + 1) / total_tests
        yield test, result, progress


def run_cli_evaluation(test_number: int = 1, top_k=5):
    """Run evaluation for a specific test (async helper for CLI)."""
    # Load tests
    tests = load_tests()

    if test_number < 0 or test_number >= len(tests):
        print(f"Error: test_row_number must be between 0 and {len(tests) - 1}")
        sys.exit(1)

    # Get the test
    test = tests[test_number]

    # Print test info
    print(f"\n{'=' * 80}")
    print(f"Test #{test_number}")
    print(f"{'=' * 80}")
    print(f"Question: {test.question}")
    print(f"Keywords: {test.keywords}")
    print(f"Category: {test.category}")
    print(f"Reference Answer: {test.reference_answer}")

    # Retrieval Evaluation
    print(f"\n{'=' * 80}")
    print("Retrieval Evaluation")
    print(f"{'=' * 80}")

    retrieval_result = evaluate_retrieval(test, top_k=top_k)

    print(f"MRR: {retrieval_result.mrr:.4f}")
    print(f"nDCG: {retrieval_result.ndcg:.4f}")
    print(
        f"Keywords Found: {retrieval_result.keywords_found}/{retrieval_result.total_keywords}"
    )
    print(f"Keyword Coverage: {retrieval_result.keyword_coverage:.1f}%")

    # Answer Evaluation
    print(f"\n{'=' * 80}")
    print("Answer Evaluation")
    print(f"{'=' * 80}")

    eval_result, generated_answer, reference_answer = evaluate_answer(test, top_k=top_k)

    print(f"\nGenerated Answer:\n{generated_answer}")
    print(f"\nFeedback:\n{eval_result.feedback}")
    print("\nScores:")
    print(f"  Accuracy: {eval_result.accuracy:.2f}/5")
    print(f"  Completeness: {eval_result.completeness:.2f}/5")
    print(f"  Relevance: {eval_result.relevance:.2f}/5")
    print(f"\n{'=' * 80}\n")


def main():
    """CLI to evaluate a specific test by row number."""
    if len(sys.argv) != 2:
        print("Usage: uv run eval.py <test_row_number>")
        sys.exit(1)

    try:
        test_number = int(sys.argv[1])
    except ValueError:
        print("Error: test_row_number must be an integer")
        sys.exit(1)

    run_cli_evaluation(test_number)


if __name__ == "__main__":
    main()
