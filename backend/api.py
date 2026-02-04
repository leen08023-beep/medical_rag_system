import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import ollama
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Medical RAG API")

# --- Configuration ---
# 'qdrant' and 'ollama' refer to the service names in docker-compose
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
COLLECTION_NAME = "medical_rag"

# Initialize Clients
q_client = QdrantClient(url=f"http://{QDRANT_HOST}:6333")
o_client = ollama.Client(host=OLLAMA_HOST)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# CORS Setup
app.add_middleware(CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str

# --- Endpoints ---

@app.post("/ingest")
async def ingest_data():
    """Reads the medical text file, chunks it, and uploads to Qdrant."""
    file_path = "data/pmc_diabetes.txt"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Medical data file not found.")

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Simple Chunking (700 chars with overlap)
    chunks = [text[i:i+700] for i in range(0, len(text), 600)]
    embeddings = embed_model.encode(chunks)

    # Setup Qdrant Collection
    q_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=len(embeddings[0]), distance=Distance.COSINE)
    )

    # Upload
    points = [
        PointStruct(id=i, vector=emb.tolist(), payload={"text": chunk})
        for i, (emb, chunk) in enumerate(zip(embeddings, chunks))
    ]
    q_client.upsert(collection_name=COLLECTION_NAME, points=points)
    
    return {"message": f"Successfully ingested {len(chunks)} chunks."}

@app.post("/chat")
async def chat(request: ChatRequest):
    """RAG: Retrieve context from Qdrant and generate answer with Ollama."""
    # 1. Search Qdrant
    query_vector = embed_model.encode(request.question).tolist()
    search_results = q_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=3
    )
    context = "\n".join([res.payload["text"] for res in search_results])

    # 2. Generate with Ollama
    prompt = f"Using this context: {context}\n\nAnswer the question: {request.question}"
    try:
        response = o_client.chat(
            model="llama3.2:3b",
            messages=[{"role": "user", "content": prompt}]
        )
        return {"answer": response["message"]["content"], "context_used": context}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))