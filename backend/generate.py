import ollama

client = ollama.Client(host="http://ollama:11434")
def generate_answer(question: str, context: str):
    prompt = f"""
You are a medical assistant.
Answer the question using ONLY the information in the context.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer in 3–5 complete sentences:
"""

    response = ollama.chat(
        model="llama3.2:3b",  
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0.2
        }
    )

    return response["message"]["content"]
