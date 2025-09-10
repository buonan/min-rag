import faiss
import openai
import tiktoken
from typing import List
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI()

# 1. Sample document (you can replace with your own text or PDF content)
document = """
RAG stands for Retrieval-Augmented Generation.
It combines information retrieval with large language models.
RAG reduces hallucination by grounding answers in external data.
It is commonly used with vector databases like FAISS or Pinecone.
"""

# 2. Split text into chunks (here we just split by sentences for simplicity)
chunks = document.split("\n")

# 3. Embed each chunk
embeddings = []
texts = []
for chunk in chunks:
    if chunk.strip():
        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        ).data[0].embedding
        embeddings.append(emb)
        texts.append(chunk)

# 4. Store embeddings in FAISS index
dimension = len(embeddings[0])
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype("float32"))

# 5. Define a query
query = "What is RAG and why is it useful?"

# Embed query
query_emb = client.embeddings.create(
    model="text-embedding-3-small",
    input=query
).data[0].embedding

# 6. Search FAISS for top matches
D, I = index.search(np.array([query_emb]).astype("float32"), k=2)
retrieved_chunks = [texts[i] for i in I[0]]

# 7. Build augmented prompt
context = "\n".join(retrieved_chunks)
prompt = f"""
You are a helpful assistant. Use the following context to answer:

Context:
{context}

Question:
{query}

Answer:
"""

# 8. Ask GPT with retrieved context
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

print("Answer:", response.choices[0].message["content"])

