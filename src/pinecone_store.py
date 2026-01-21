from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.config import PINECONE_API_KEY, PINECONE_INDEX


def get_vectorstore(docs, embeddings):
    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing_indexes = [idx.name for idx in pc.list_indexes()]
    
    if PINECONE_INDEX not in existing_indexes:
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX,
        embedding=embeddings,
    )

    vectorstore.add_documents(docs)

    return vectorstore
