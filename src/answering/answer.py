import torch
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document

from dotenv import load_dotenv


load_dotenv(override=True)

DEBUG = os.getenv("DEBUG", "False").lower() == "true"

KNOWLEDGE_BASE_DIR = str(Path(__file__).parent.parent.parent / "knowledge-base")
VECTOR_DB_NAME = str(Path(__file__).parent.parent.parent / "vector-db")
COLLECTION_NAME = "docs"

hg_embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

openai_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", 
    model_kwargs={
        "temperature": 0.0, 
        "max_tokens": 2048, 
        "top_p": 1.0,
        }
)