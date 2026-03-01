import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# 1. Embedding model (fast & local)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Sample document
document = """
RAG stands for Retrieval-Augmented Generation.
It combines information retrieval with large language models.
RAG reduces hallucination by grounding answers in external data.
It is commonly used with vector databases like FAISS or Pinecone.
"""
chunks = [c.strip() for c in document.split("\n") if c.strip()]

# 3. Embed chunks
embeddings = embedder.encode(chunks, convert_to_numpy=True)

# 4. Store in FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# 5. Query
query = "What is RAG and why is it useful?"
# query = "Whos is Bill Gates mother?"

query_emb = embedder.encode([query], convert_to_numpy=True)

D, I = index.search(query_emb, k=2)
retrieved_chunks = [chunks[i] for i in I[0]]

# 6. Build augmented prompt
context = "\n".join(retrieved_chunks)
prompt = f"""
Answer the question using the provided context.

Context:
{context}

Question:
{query}

Answer:
"""

# 7. Hugging Face LLM (Flan-T5, works offline, no login required)
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"  # ✅ smaller & open
)

response = generator(prompt, max_new_tokens=200)
print("Answer:", response[0]["generated_text"])

