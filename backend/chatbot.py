from retrieve import retrieve
from generate import generate_answer

print("Medical RAG Chatbot")
print("Type 'exit' or 'quit' to stop.\n")


history = []

while True:
    query = input("You: ").strip()
    
    if query.lower() in ["exit", "quit"]:
        print("Exiting chatbot. Goodbye!")
        break

   
    contexts = retrieve(query, k=5)
    context_text = "\n".join(contexts)

    
    memory_text = "\n".join(history[-6:])  

    prompt_context = f"{memory_text}\n{context_text}" if memory_text else context_text

    
    answer = generate_answer(query, prompt_context)

   
    print("\nAssistant:", answer)
    print("-" * 50)

    
    history.append(f"You: {query}")
    history.append(f"Assistant: {answer}")
