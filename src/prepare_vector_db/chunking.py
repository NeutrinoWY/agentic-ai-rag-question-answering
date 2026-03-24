from typing import List, Optional
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
from tenacity import retry, wait, stop_after_attempt, wait_exponential
from multiprocessing import Pool
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from src.utils.pydantic_models import Chunks
from src.utils.prompts import chunking_prompt

load_dotenv(override=True)


def create_messages(document: dict) -> dict:
    """Create a message for the LLM to split the document into chunks.
    Args:
        prompt: The prompt to be sent to the LLM.
    returns: A message dict for the LLM.
    """
    return {"role": "user", "content": chunking_prompt(document)}


# @retry(wait=wait)
def chunk_document_llm(
    document: dict, chunking_llm: str = "gpt-4.1-nano"
) -> List[dict]:
    """Chunk a single document into smaller chunks using an LLM-based approach.
    Args:
        document: The document to be split into chunks.
        chunking_llm: The LLM model to use for chunking the document.
    returns: A list of chunks for the document, where each chunk is a dict containing a headline, the original text of the chunk, and metadata from the original document.
    """
    messages = create_messages(document)
    # use pydantic model Chunks to validate the response, chunks is a list of dicts with the keys "headline", "text"
    # chunk the document using the LLM, with a response format of Chunks
    # todo: initialize client inside create_chunks_llm if using parallel processing
    response = OpenAI().chat.completions.parse(
        model=chunking_llm,
        messages=[messages],
        response_format=Chunks,
        temperature=0,  # lower temperature for more deterministic output
        # max_tokens=3000,  # Not to add in case long document
    )
    reply = response.choices[0].message.content

    # check and parse the reply into a list of Chunk objects
    doc_as_chunks = Chunks.model_validate_json(reply).chunks

    # convert the list of Chunk objects into a list of dicts with the original document's metadata
    return [chunk.as_result(document) for chunk in doc_as_chunks]


def create_chunks_llm(
    documents: List, chunking_llm: str = "gpt-4.1-nano", workers: Optional[int] = 4
) -> List[dict]:
    """Split documents into smaller chunks using an LLM-based approach.
    Each chunk is a dict containing a headline, the original text of the chunk, and metadata from the original document.
    Args:
        documents: List of documents to be split into chunks.
        chunking_llm: model name
        workers: number of workers
    returns: List of document chunks with metadata.
    """
    chunks = []

    if len(documents) > 10:
        # [may have API limits] We use imap_unordered to process documents in parallel and get results as they come in,
        # which is more efficient than waiting for all to finish.
        with Pool(processes=workers) as pool:
            for result in tqdm(
                pool.imap_unordered(chunk_document_llm, documents), total=len(documents)
            ):
                # result is a list of chunks for a single document
                chunks.extend(result)
    else:
        for document in tqdm(documents, desc="Chunking documents with LLM"):
            chunks.extend(chunk_document_llm(document, chunking_llm=chunking_llm))
    return chunks


def chunk_document_emb(
    document: dict, chunking_emb: str = "text-embedding-3-small"
) -> List[dict]:
    """Split a single document into smaller chunks using an embedding-based approach with LangChain's SemanticChunker.
    Each chunk is a dict containing the original text of the chunk and metadata from the original document.
    Args:
        document: The document to be split into chunks.
        chunking_emb: The embedding model to use for chunking (default is "text-embedding-3-small").
    returns: List of document chunks with metadata.
    """
    # Initialize the text splitter with the specified embedding model
    # Todo: if not use parallel processing, initialize model in main chunking function to avoid repeated initialization
    embeddings = OpenAIEmbeddings(model=chunking_emb)
    text_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",  # use percentile to determine breakpoints based on the distribution of embedding distances to structure the chunks more effectively
        breakpoint_threshold_amount=95,  # set the threshold to the 95th percentile to allow for more natural chunking while still keeping chunks reasonably sized
    )

    chunks = text_splitter.split_text(document.page_content)
    chunk_docs = []
    for i, chunk in enumerate(chunks):
        chunk_doc = {
            "text": chunk,
            "metadata": {
                "type": document.metadata.get("type", "unknown"),
                "source": document.metadata.get("source", ""),
                "headline": f"{document.metadata.get('type', 'unknown')}_chunk_{i}",
            },
        }
        chunk_docs.append(chunk_doc)
    return chunk_docs


def create_chunks_emb(
    documents: List,
    chunking_emb: str = "text-embedding-3-small",
    workers: Optional[int] = 4,
) -> List[dict]:
    """Split documents into smaller chunks using an embedding-based approach with LangChain's SemanticChunker.
    Each chunk is a dict containing the original text of the chunk and metadata from the original document.
    Args:
        documents: List of documents to be split into chunks.
        chunking_emb: The embedding model to use for chunking (default is "text-embedding-3-small").
    returns: List of document chunks with metadata.
    """
    all_chunks = []
    if len(documents) > 10:
        # [may have API limits and memory constraints]
        with Pool(processes=workers) as pool:
            for result in tqdm(
                pool.imap_unordered(chunk_document_emb, documents), total=len(documents)
            ):
                # result is a list of chunks for a single document
                all_chunks.extend(result)
    else:
        for document in tqdm(
            documents, desc="Chunking documents with embedding-based approach"
        ):
            all_chunks.extend(chunk_document_emb(document, chunking_emb=chunking_emb))
    return all_chunks


def create_chunks(documents: List, chunk_size: int = 500, chunk_overlap: int = 150):
    """Split documents into smaller chunks using RecursiveCharacterTextSplitter.
    Each chunk will retain the metadata of its parent document, including the "type" field.
    Args:
        documents: List of documents to be split into chunks.
        chunk_size: The maximum size of each chunk.
        chunk_overlap: The number of characters to overlap between chunks.
    returns: List of document chunks with metadata.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
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
                },
            }
            all_chunks.append(chunk_doc)
    return all_chunks
