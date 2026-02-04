from retrieve import retrieve
from generate import generate_answer

query = "How is diabetes treated?"

contexts = retrieve(query)
context_text = "\n".join(contexts)

answer = generate_answer(query, context_text)

print("Question:", query)
print("Answer:", answer)
