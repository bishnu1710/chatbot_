from rag_utils import generate_response

query = "I'm feeling very anxious lately."
res = generate_response(query, history=[])
print(f"User: {query}")
print(f"AI: {res['answer']}")
