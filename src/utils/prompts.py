
from typing import List, Optional, Tuple


def chunking_prompt(document: dict) -> str:
    """Create a prompt for the LLM to split the document into chunks.
    Args:
        document: The document to be split.
    returns: A prompt string for the LLM.
    """
    how_many = max(5, len(document.page_content) // 500)  # heuristic for number of chunks
    prompt = f"""
You take a document and you split the document into overlapping chunks for a KnowledgeBase.

The document is from the shared drive of a company called Insurellm.
The document is of type: {document.metadata.get("type", "unknown")}.

A chatbot will use these chunks to answer questions about the company.
You should divide up the document as you see fit, being sure that the entire document is returned across the chunks - don't leave anything out.
This document should probably be split into at least {how_many} chunks, but you can have more or less as appropriate, ensuring that there are individual chunks to answer specific questions.
There should be overlap between the chunks as appropriate; typically about 25% overlap or about 50 words, so you have the same text in multiple chunks for best retrieval results.

For each chunk, you should provide a headline, a summary, and the original text of the chunk.
Together your chunks should represent the entire document with overlap.

Here is the document:

{document.page_content}

Respond with the chunks.
"""
    return prompt



def system_prompt(
        context: str
        ) -> str:
    """Create a prompt for the LLM to answer the question using the retrieved chunks as context.
    Args:
        question: The user's question to be answered.
        context: The context containing the retrieved chunks.
    returns: A prompt string for the LLM.
    """
    prompt = f"""You are a knowledgable, polite, and concise assistant for answering questions about the compnay. 
    If relevant information is available in the context, use it to answer the question.
    If the context does not contain relevant information, say you don't know.
    Here is the context:
    {context}
    """
    return prompt



def evaluation_prompt(
        question: str,
        reference_answer: str,
        answer: str,
) -> Tuple[str, str]:

    eval_system_prompt = """
    You are an expert evaluator assessing the quality of answers. 
    Evaluate the generated answer by comparing it to the reference answer. 
    Only give 5/5 scores for perfect answers.
    """

    eval_user_prompt = f"""Question:
            {question}

            Reference Answer:
            {reference_answer}

            Generated Answer:
            {answer}

            Please evaluate the generated answer on three dimensions:
            1. Accuracy: How factually correct is it compared to the reference answer? Only give 5/5 scores for perfect answers.
            2. Completeness: How thoroughly does it address all aspects of the question, covering all the information from the reference answer?
            3. Relevance: How well does it directly answer the specific question asked, giving no additional information?

            Provide detailed feedback and scores from 1 (very poor) to 5 (ideal) for each dimension. If the answer is wrong, then the accuracy score must be 1.
            """
    return  eval_system_prompt, eval_user_prompt