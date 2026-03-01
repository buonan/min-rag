import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from pypdf import PdfReader

# -----------------------------
# 1. Load and extract text from PDF
# -----------------------------
def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# Example: replace with your PDF file path
pdf_text = load_pdf("/home/brian/workspace/rag-ai/min-rag/example.pdf")

# -----------------------------
# 2. Chunk the document
# -----------------------------
def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

chunks = chunk_text(pdf_text)

# -----------------------------
# 3. Embeddings with Hugging Face
# -----------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(chunks, convert_to_numpy=True)

# -----------------------------
# 4. Store in FAISS
# -----------------------------
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# -----------------------------
# 5. Query
# -----------------------------
query = "Summarize the key ideas from this PDF."
query_emb = embedder.encode([query], convert_to_numpy=True)

D, I = index.search(query_emb, k=3)
retrieved_chunks = [chunks[i] for i in I[0]]

# -----------------------------
# 6. Build augmented prompt
# -----------------------------
context = "\n".join(retrieved_chunks)
prompt = f"""
Answer the question using the provided context.

Context:
{context}

Question:
{query}

Answer:
"""

# -----------------------------
# 7. Hugging Face LLM for generation
# -----------------------------
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"  # smaller CPU-friendly model
)

response = generator(prompt, max_new_tokens=200)
print("Answer:", response[0]["generated_text"])

