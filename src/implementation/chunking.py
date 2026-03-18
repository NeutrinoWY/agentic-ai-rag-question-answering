import os
import glob
from pathlib import Path
from typing import List, Optional
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm
from tenacity import retry, wait, stop_after_attempt, wait_exponential
from multiprocessing import Pool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic_models import Chunks


load_dotenv(override=True)


def create_prompt(document: dict) -> str:
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


def create_messages(document: dict) -> dict:
    """Create a message for the LLM to split the document into chunks.
    Args:
        prompt: The prompt to be sent to the LLM.
    returns: A message dict for the LLM.
    """
    return {
        "role": "user",
        "content": create_prompt(document)
    }



# @retry(wait=wait)
def chunk_document_llm(document: dict, chunking_llm: str = "gpt-4.1-nano"):
    messages = create_messages(document)
    # use pydantic model Chunks to validate the response, chunks is a list of dicts with the keys "headline", "text"
    # chunk the document using the LLM, with a response format of Chunks
    response = OpenAI().chat.completions.parse(
        model=chunking_llm,
        messages=[messages],
        response_format=Chunks,
        temperature=0,  # lower temperature for more deterministic output
        max_tokens=3000,  # adjust as needed based on expected chunk size and number of chunks
    )
    reply = response.choices[0].message.content

    # check and parse the reply into a list of Chunk objects
    doc_as_chunks = Chunks.model_validate_json(reply).chunks

    # convert the list of Chunk objects into a list of dicts with the original document's metadata
    return [chunk.as_result(document) for chunk in doc_as_chunks]



def create_chunks_llm(documents: List, chunking_llm: str = "gpt-4.1-nano"):
    """Split documents into smaller chunks using an LLM-based approach.
    Each chunk is a dict containing a headline, the original text of the chunk, and metadata from the original document.
    Args:
        documents: List of documents to be split into chunks.
        prompt: Optional prompt for the LLM.
    returns: List of document chunks with metadata.
    """
    chunks = []

    if len(documents) > 10:
        # We use imap_unordered to process documents in parallel and get results as they come in, 
        # which is more efficient than waiting for all to finish.
        workers = int(os.getenv("WORKERS", 4)) # should be integer
        with Pool(processes=workers) as pool:
            for result in tqdm(pool.imap_unordered(chunk_document_llm, documents), total=len(documents)):
                # result is a list of chunks for a single document
                chunks.extend(result)
    else:
        for document in tqdm(documents, desc="Chunking documents with LLM"):
            chunks.extend(chunk_document_llm(document, chunking_llm=chunking_llm))
    return chunks




def create_chunks(documents: List):
    """Split documents into smaller chunks using RecursiveCharacterTextSplitter.
    Each chunk will retain the metadata of its parent document, including the "type" field.
    Args:
        documents: List of documents to be split into chunks.
    returns: List of document chunks with metadata.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
    # chunks = text_splitter.split_documents(documents)

    # split each document into chunks and retain metadata
    all_chunks = []
    for doc in tqdm(documents, desc="Splitting documents into chunks"):
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            chunk_doc = {
                "text": chunk,
                "metadata": {
                    "type": doc.metadata.get("type", "unknown"),
                    "source": doc.metadata.get("source", ""),
                    "headline": f"{doc.metadata.get('type', 'unknown')}_chunk_{i}",
                }
            }
            all_chunks.append(chunk_doc)
    return all_chunks




