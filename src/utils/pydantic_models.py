from pydantic import BaseModel, Field
from typing import List


class Chunk(BaseModel):
    """A knowledge base chunk from documents in the knowledge base."""

    headline: str = Field(
        ..., description="A headline summarizing the content of the chunk"
    )
    text: str = Field(..., description="The original text of the chunk")

    def as_result(self, document: dict) -> dict:
        """Convert the chunk into a result format that includes the original document's metadata.
        Args:
            document: The original document from which the chunk was created.
        returns: A dict containing the chunk's headline, summary, text, and the original document's metadata.
        """
        return {
            "text": self.text,
            "metadata": {
                "type": document.metadata.get("type", "unknown"),
                "source": document.metadata.get("source", ""),
                "headline": self.headline,
            },
        }


class Chunks(BaseModel):
    """List of knowledge base chunks"""

    chunks: List[Chunk]


class Answer(BaseModel):
    """Model response for a user question."""

    answer: str = Field(description="The detailed answer to the question")
    source: str = Field(
        description="The original text in context that support the answer"
    )


class RetrievalEval(BaseModel):
    """Evaluation metrics for retrieval performance."""

    mrr: float = Field(description="Mean Reciprocal Rank - average across all keywords")
    ndcg: float = Field(
        description="Normalized Discounted Cumulative Gain (binary relevance)"
    )
    keywords_found: int = Field(description="Number of keywords found in top-k results")
    total_keywords: int = Field(description="Total number of keywords to find")
    keyword_coverage: float = Field(description="Percentage of keywords found")


class AnswerEval(BaseModel):
    """LLM-as-a-judge evaluation of answer quality when comparing with test dataset."""

    feedback: str = Field(
        description="Concise feedback on the answer quality, comparing it to the reference answer and evaluating based on the retrieved context"
    )
    accuracy: float = Field(
        description="How factually correct is the answer compared to the reference answer? 1 (wrong. any wrong answer must score 1) to 5 (ideal - perfectly accurate). An acceptable answer would score 3."
    )
    completeness: float = Field(
        description="How complete is the answer in addressing all aspects of the question? 1 (very poor - missing key information) to 5 (ideal - all the information from the reference answer is provided completely). Only answer 5 if ALL information from the reference answer is included."
    )
    relevance: float = Field(
        description="How relevant is the answer to the specific question asked? 1 (very poor - off-topic) to 5 (ideal - directly addresses question and gives no additional information). Only answer 5 if the answer is completely relevant to the question and gives no additional information."
    )
