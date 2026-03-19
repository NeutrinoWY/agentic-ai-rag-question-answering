

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




