from langchain_openai import OpenAIEmbeddings
from src.config import OPENROUTER_API_KEY

def get_embeddings():
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )
