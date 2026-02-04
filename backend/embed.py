from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from ingest import load_documents

import os
# This allows the code to work both in Docker and locally
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost") 
client = QdrantClient(url=f"http://{QDRANT_HOST}:6333")
docs = load_documents()
print("Number of chunks:", len(docs))


model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(docs)


# Change this:
# client = QdrantClient(url="http://localhost:6333") 

# To this:

collection_name = "medical_rag"


client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=embeddings.shape[1],
        distance=Distance.COSINE
    )
)


points = [
    {
        "id": i,
        "vector": embeddings[i].tolist(),
        "payload": {"text": docs[i]}
    }
    for i in range(len(docs))
]

client.upsert(
    collection_name=collection_name,
    points=points
)

print("✅ Embeddings uploaded to Qdrant successfully!")


