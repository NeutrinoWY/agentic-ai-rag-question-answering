from pydantic import BaseModel, Field
from typing import List

class Chunk(BaseModel):
    headline: str = Field(..., description="A headline summarizing the content of the chunk")
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
            }
        }

class Chunks(BaseModel):
    chunks: List[Chunk]


class Answer(BaseModel):
    answer: str = Field(description="The detailed answer to the question")
    source: str = Field(description="The original text in context that support the answer")