from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.config import PINECONE_API_KEY, PINECONE_INDEX


def get_vectorstore(docs, embeddings):
    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing_indexes = [idx.name for idx in pc.list_indexes()]
    
    if PINECONE_INDEX not in existing_indexes:
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
    else:
        # Verify dimension matches
        index_description = pc.describe_index(PINECONE_INDEX)
        actual_dim = index_description.dimension
        if actual_dim != 384:
            raise ValueError(
                f"Dimension mismatch! Index '{PINECONE_INDEX}' has dimension {actual_dim}, "
                f"but the new embedding model requires 384. "
                f"Please delete the index in Pinecone or change the index name in your .env file."
            )
        
    index = pc.Index(PINECONE_INDEX)
    index.delete(delete_all=True)

    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX,
        embedding=embeddings,
    )

    vectorstore.add_documents(docs)

    return vectorstore
