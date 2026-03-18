import os
import glob
from pathlib import Path
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from typing import List, Optional
from dotenv import load_dotenv
import torch
from chunking import create_chunks, create_chunks_llm


load_dotenv(override=True)

DEBUG = os.getenv("DEBUG", "False").lower() == "true"

KNOWLEDGE_BASE_DIR = str(Path(__file__).parent.parent.parent / "knowledge-base")
VECTOR_DB_NAME = str(Path(__file__).parent.parent.parent / "vector-db")
COLLECTION_NAME = "docs"

hg_embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2", 
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"})

openai_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", 
    model_kwargs={
        "temperature": 0.0, 
        "max_tokens": 2048, 
        "top_p": 1.0,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
        })



def fetch_documents():
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
                loader_kwargs={"encoding": "utf-8"}
            )
            docs = loader.load()  # Load documents from the folder
            for doc in docs:
                # doc format is {"page_content": ..., "metadata": {"source": ...}}
                doc.metadata["type"] = doc_type
                documents.append(doc)
    # format of a document is {"page_content": ..., "metadata": {"type": ..., "source": ...}}
    return documents


def fetch_documents_simple():
    """Fetch all documents from the knowledge base directory, categorizing them by type."""

    documents = []
    for folder in KNOWLEDGE_BASE_DIR.iterdir():
        doc_type = folder.name
        for file in folder.rglob("*.md"):
            with open(file, "r", encoding="utf-8") as f:
                # keep the same format as the documents loaded by DirectoryLoader
                documents.append({"metadata": {"type": doc_type, "source": file.as_posix()}, 
                                  "page_content": f.read()})
    print(f"Loaded {len(documents)} documents")
    return documents



def generate_chunks(documents: List):
    """Split documents into smaller chunks using RecursiveCharacterTextSplitter or LLM.
    Each chunk will retain the metadata of its parent document, including the "type" field.
    Args:
        documents: List of documents to be split into chunks.
    returns: List of document chunks with metadata.
    """
    if os.getenv("LLM_CHUNKING", "False").lower() == "true":
        return create_chunks_llm(documents)
    else:
        return create_chunks(documents)


def create_vector_db_via_langchain(chunks: list, embedding_model: Optional[str] = "huggingface"):
    """Create a Chroma vector database from the provided document chunks.
    Each chunk's content will be embedded using a embedding model and stored in the vector database along with its metadata.
    Args:
        chunks: List of document chunks to be embedded and stored in the vector database.
        embedding_model: The embedding model to use for embedding the chunks.
    returns: A Chroma vector database instance containing the embedded chunks.
    """
    # Create a Chroma vector database using the specified embedding model
    if embedding_model == "huggingface":
        embedding = hg_embeddings
    elif embedding_model == "openai":
        embedding = openai_embeddings
    else:
        raise ValueError("Unsupported embedding model")
    
    # delete existing collection if the vector db exist and the collection exists to avoid duplicates
    if os.path.exists(VECTOR_DB_NAME):
        vector_db = Chroma(
            collection_name=COLLECTION_NAME, 
            persist_directory=VECTOR_DB_NAME,
            embedding_function=embedding
            )
        vector_db.delete_collection()

    vector_db = Chroma.from_texts(
        embedding=embedding,
        texts=[chunk["text"] for chunk in chunks],
        metadatas=[chunk["metadata"] for chunk in chunks],
        collection_name=COLLECTION_NAME,
        persist_directory=VECTOR_DB_NAME
    )
    # The format of a chunk is {"text": ..., "metadata": {"type": ..., "source": ..., "headline": ...}}

    return vector_db



def ingest_knowledge_base():
    documents = fetch_documents() 
    # print the first document's content and metadata to verify
    if documents:
        print(f"First document content: {documents[0].page_content[:500]}")  # print first 500 characters
        print(f"First document metadata: {documents[0].metadata}")
    
    llm_chunking = os.getenv("LLM_CHUNKING", "False")
    if DEBUG or llm_chunking.lower() == "true":
        print("Using LLM-based chunking...")
        documents = documents[:1]  # Limit to first 2 documents for LLM chunking to save time
    
    if llm_chunking.lower() == "true":
        # use LLM-based chunking for better quality chunks, especially for complex documents, but it is slower and more expensive. We limit the number of documents to 1 in debug mode to save time and cost.
        chunks = create_chunks_llm(documents, chunking_llm="gpt-4.1-nano")
    else:
        chunks = create_chunks(documents)

    create_vector_db_via_langchain(chunks, embedding_model="huggingface")


if __name__ == "__main__":
    ingest_knowledge_base()

    # check database contents
    vector_db = Chroma(collection_name=COLLECTION_NAME, persist_directory=VECTOR_DB_NAME)
    print(f"Number of documents in the vector database: {vector_db._collection.count()}")
    sample_embedding = vector_db._collection.get(limit=1, include=["embeddings", "metadatas"])
    if sample_embedding:
        print(f"Sample embedding: {sample_embedding}")
