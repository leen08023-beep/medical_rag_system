from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter
import os
# This allows the code to work both in Docker and locally
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost") 
client = QdrantClient(url=f"http://{QDRANT_HOST}:6333")


COLLECTION_NAME = "medical_rag"

# Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve(query: str, k: int = 5):
    query_vector = model.encode(query).tolist()

    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=k,
        with_payload=True
    )

    return [point.payload["text"] for point in result.points]
