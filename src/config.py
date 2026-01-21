import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

if not PINECONE_INDEX:
    raise ValueError(
        "PINECONE_INDEX is not set. "
        "Check your .env file and ensure PINECONE_INDEX=swiggy-rag"
    )
