import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from pypdf import PdfReader
import glob

# -----------------------------
# 1. Load and chunk PDFs
# -----------------------------
def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

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

# Load all PDFs from folder
pdf_files = glob.glob("pdfs/example/*.pdf")  # folder with PDFs
all_chunks = []
for pdf_file in pdf_files:
    text = load_pdf(pdf_file)
    chunks = chunk_text(text)
    all_chunks.extend(chunks)

print(f"Loaded {len(all_chunks)} chunks from {len(pdf_files)} PDFs.")

# -----------------------------
# 2. Embeddings
# -----------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(all_chunks, convert_to_numpy=True)

# -----------------------------
# 3. FAISS index
# -----------------------------
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# -----------------------------
# 4. Hugging Face LLM
# -----------------------------
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
)

# -----------------------------
# 5. Interactive loop
# -----------------------------
print("\n=== Multi-PDF RAG System ===")
print("Type your question and press Enter. Type 'exit' to quit.\n")

while True:
    query = input("Your question: ")
    if query.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    # Embed query and retrieve top chunks
    query_emb = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(query_emb, k=5)
    retrieved_chunks = [all_chunks[i] for i in I[0]]

    # Build prompt for LLM
    context = "\n".join(retrieved_chunks)
    prompt = f"""
Answer the question using the provided context.

Context:
{context}

Question:
{query}

Answer:
"""
    # Generate answer
    response = generator(prompt, max_new_tokens=300)
    print("\nAnswer:", response[0]["generated_text"])
    print("-" * 80)

