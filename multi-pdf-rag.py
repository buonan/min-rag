import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from pypdf import PdfReader
import glob

# -----------------------------
# 1. Load text from a PDF
# -----------------------------
def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# -----------------------------
# 2. Chunk text
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

# -----------------------------
# 3. Load multiple PDFs
# -----------------------------
pdf_files = glob.glob("pdfs/realestate/*.pdf")  # folder with PDFs
all_chunks = []

for pdf_file in pdf_files:
    text = load_pdf(pdf_file)
    chunks = chunk_text(text)
    all_chunks.extend(chunks)

print(f"Total chunks loaded: {len(all_chunks)}")

# -----------------------------
# 4. Embed all chunks
# -----------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(all_chunks, convert_to_numpy=True)

# -----------------------------
# 5. Store in FAISS
# -----------------------------
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# -----------------------------
# 6. Query
# -----------------------------
#query = "Summarize the key ideas from all PDFs."
query = "What are the main factors affecting real estate prices according to these documents?"
query_emb = embedder.encode([query], convert_to_numpy=True)

D, I = index.search(query_emb, k=5)  # retrieve top 5 chunks
retrieved_chunks = [all_chunks[i] for i in I[0]]

# -----------------------------
# 7. Build prompt
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
# 8. Hugging Face LLM fo

# 7. Hugging Face LLM (Flan-T5, works offline, no login required)
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"  # ✅ smaller & open
)

response = generator(prompt, max_new_tokens=200)
print("Answer:", response[0]["generated_text"])


