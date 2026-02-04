# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins (for dev)
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    message: str

@app.post("/chat")
async def chat(msg: Message):
    user_input = msg.message
    # Call your RAG system here
    # Example:
    bot_reply = f"Echo: {user_input}"  # Replace with RAG response
    return {"reply": bot_reply}
