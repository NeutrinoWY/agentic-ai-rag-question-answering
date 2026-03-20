import os
import glob
import yaml
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from src.prepare_vector_db.chunking import (
    create_chunks,
    create_chunks_llm,
    create_chunks_emb,
)
from src.utils.utils import CONFIG

from dotenv import load_dotenv

load_dotenv(override=True)


PROJECT_ROOT = Path(__file__).parent.parent.parent
KNOWLEDGE_BASE_DIR = str(PROJECT_ROOT / CONFIG["knowledge-base"])    
VECTOR_DB_NAME = str(PROJECT_ROOT / CONFIG["vectorDB"]["name"])        
COLLECTION_NAME = CONFIG["vectorDB"]["collection_name"]


def fetch_documents() -> List[dict]:
    """Fetch all documents from the knowledge base directory, categorizing them by type.
    Each document's metadata will include a "type" field corresponding to its parent folder name.
    returns: List of documents with metadata including their type.
    """
    folders = glob.glob(KNOWLEDGE_BASE_DIR + "/*")
    documents = []
    for folder in folders:
        if os.path.isdir(folder):
            doc_type = os.path.basename(folder)
            loader = DirectoryLoader(
                folder,
                glob="**/*.md",
                show_progress=True,
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"},
            )
            docs = loader.load()  # Load documents from the folder
            for doc in docs:
                # doc format is {"page_content": ..., "metadata": {"source": ...}}
                doc.metadata["type"] = doc_type
                documents.append(doc)
    # format of a document is {"page_content": ..., "metadata": {"type": ..., "source": ...}}
    return documents


def fetch_documents_simple() -> List[dict]:
    """Fetch all documents from the knowledge base directory, categorizing them by type."""

    documents = []
    for folder in KNOWLEDGE_BASE_DIR.iterdir():
        doc_type = folder.name
        for file in folder.rglob("*.md"):
            with open(file, "r", encoding="utf-8") as f:
                # keep the same format as the documents loaded by DirectoryLoader
                documents.append(
                    {
                        "metadata": {"type": doc_type, "source": file.as_posix()},
                        "page_content": f.read(),
                    }
                )
    print(f"Loaded {len(documents)} documents")
    return documents


def generate_chunks(
    chunking_method: str,
    documents: List,
    chunk_size: Optional[int] = 500,
    chunk_overlap: Optional[int] = 150,
    chunking_emb: Optional[str] = "text-embedding-3-small",
    chunking_llm: Optional[str] = "gpt-4.1-nano",
    workers: Optional[int] = 4,
) -> List[dict]:
    """Split documents into smaller chunks using RecursiveCharacterTextSplitter or LLM.
    Each chunk will retain the metadata of its parent document, including the "type" field.
    Args:
        chunking_method: The method to use for chunking (e.g., "recursive", "llm", "embedding").
        documents: List of documents to be split into chunks.
        chunk_size: The maximum size of each chunk.
        chunk_overlap: The number of characters to overlap between chunks.
        chunking_emb: The embedding model to use for embedding-based chunking (default is "text-embedding-3-small").
        chunking_llm: The LLM model to use for LLM-based chunking (default is "gpt-4.1-nano").
        workers: The number of worker processes to use for parallel chunking (default is 4).
    returns: List of document chunks with metadata.
    """
    # use open ai llm to chunk
    if chunking_method == "llm":
        return create_chunks_llm(documents, chunking_llm=chunking_llm, workers=workers)
    # use SemanticChunker with openai embedding model in langchain to chunk
    elif chunking_method == "embedding":
        return create_chunks_emb(documents, chunking_emb=chunking_emb, workers=workers)
    # use recursivetextsplit in langchain to chunk
    else:
        return create_chunks(
            documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )


def create_vector_db(
    chunks: list, embedding_method: Optional[str] = "huggingface"
) -> Chroma:
    """Create a Chroma vector database from the provided document chunks.
    Each chunk's content will be embedded using a embedding model and stored in the vector database along with its metadata.
    Args:
        chunks: List of document chunks to be embedded and stored in the vector database.
        embedding_method: The embedding method to use for embedding the chunks.
    returns: A Chroma vector database instance containing the embedded chunks.
    """
    # Create a Chroma vector database using the specified embedding model
    if embedding_method == "huggingface":
        embedding = HuggingFaceEmbeddings(model_name=CONFIG["vectorDB"]["hf_model"])

    elif embedding_method == "openai":
        embedding = OpenAIEmbeddings(
            model=CONFIG["vectorDB"]["openai_model"]
        )
    else:
        raise ValueError("Unsupported embedding method")

    # delete existing collection if the vector db exist and the collection exists to avoid duplicates
    if os.path.exists(VECTOR_DB_NAME):
        vector_db = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=VECTOR_DB_NAME,
            embedding_function=embedding,
        )
        vector_db.delete_collection()

    vector_db = Chroma.from_texts(
        embedding=embedding,
        texts=[chunk["text"] for chunk in chunks],
        metadatas=[chunk["metadata"] for chunk in chunks],
        collection_name=COLLECTION_NAME,
        persist_directory=VECTOR_DB_NAME,
    )
    # before ingesting, the content of a chunk is in the "text" key, and the metadata is in the "metadata" key
    # the format of a chunk is {"text": ..., "metadata": {"type": ...,  "source": ..., "headline": ...}}

    # after ingesting, the content of a chunk is in the "page_content" key, and the metadata is in the "metadata" key
    # The format of a chunk is {"page_content": ..., "metadata": {"type": ..., "source": ..., "headline": ...}}
    return vector_db


def ingest_knowledge_base(
    chunking_method: str = "llm",
    chunk_size: Optional[int] = 500,
    chunk_overlap: Optional[int] = 150,
    chunking_emb: Optional[str] = "text-embedding-3-small",
    chunking_llm: Optional[str] = "gpt-4.1-nano",
    workers: Optional[int] = 4,
    embedding_method: Optional[str] = "huggingface",
) -> None:
    """Ingest the knowledge base by fetching documents, generating chunks, and creating a vector database
    Args:
        chunking_method: The method to use for chunking (e.g., "recursive", "llm", "embedding").
        chunk_size: The maximum size of each chunk (used for recursive chunking).
        chunk_overlap: The number of characters to overlap between chunks (used for recursive chunking).
        chunking_emb: The embedding model to use for embedding-based chunking (default is "text-embedding-3-small").
        chunking_llm: The LLM model to use for LLM-based chunking (default is "gpt-4.1-nano").
        workers: The number of worker processes to use for parallel chunking (default is 4).
        embedding_method: The embedding method to use for creating the vector database (default is "huggingface").
    returns: None
    """
    documents = fetch_documents()
    # print the first document's content and metadata to verify
    if CONFIG["debug"]:
        documents = documents[
            :1
        ]  # Limit to first 2 documents for LLM chunking to save time
        print(
            f"First document content: {documents[0].page_content[:20]}"
        )  # print first 500 characters
        print(f"First document metadata: {documents[0].metadata}")

    chunks = generate_chunks(
        chunking_method=chunking_method,
        documents=documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunking_emb=chunking_emb,
        chunking_llm=chunking_llm,
        workers=workers,
    )

    create_vector_db(chunks, embedding_method=embedding_method)


if __name__ == "__main__":
    ingest_knowledge_base(
        chunking_method=CONFIG["chunking"]["chunking_method"],  # "recursive", "llm", or "embedding"
        chunk_size=CONFIG["chunking"]["chunk_size"],
        chunk_overlap=CONFIG["chunking"]["chunk_overlap"],
        chunking_emb=CONFIG["chunking"]["chunking_emb"],
        chunking_llm=CONFIG["chunking"]["chunking_llm"],
        workers=CONFIG["chunking"]["workers"],
        embedding_method=CONFIG["vectorDB"]["embedding_method"],  # "huggingface" or "openai"
    )

    # check database contents
    vector_db = Chroma(
        collection_name=COLLECTION_NAME, persist_directory=VECTOR_DB_NAME
    )
    print(
        f"Number of documents in the vector database: {vector_db._collection.count()}"
    )
    sample_embedding = vector_db._collection.get(
        limit=1, include=["embeddings", "metadatas"]
    )
    print(len(sample_embedding["embeddings"][0]))
    if sample_embedding:
        print(f"Sample embedding: {sample_embedding}")
