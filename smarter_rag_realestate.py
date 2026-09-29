#!/usr/bin/env python3

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

import faiss
import numpy as np
import requests
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder


# ============================================================
# Configuration
# ============================================================

DEFAULT_PDF_DIR = "pdfs/realestate"

DEFAULT_LLAMA_URL = (
    "http://localhost:8080/v1/chat/completions"
)

DEFAULT_HEALTH_URL = (
    "http://localhost:8080/health"
)

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L12-v2"
)

RERANKER_MODEL = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

DEFAULT_CHUNK_SIZE = 300
DEFAULT_OVERLAP = 50

DEFAULT_CANDIDATE_K = 20
DEFAULT_FINAL_K = 6

DEFAULT_MAX_TOKENS = 500


# ============================================================
# Data model
# ============================================================

@dataclass
class Chunk:
    text: str
    source: str
    page: int
    chunk_id: int
    document_type: str
    year: int | None = None


# ============================================================
# Document classification
# ============================================================

def classify_document(filename: str) -> str:

    name = filename.lower()

    if "nwmls" in name:
        return "nwmls"

    if "form" in name:
        return "nwmls"

    if "law" in name:
        return "washington_law"

    if "rcw" in name:
        return "washington_law"

    if "wac" in name:
        return "washington_law"

    if "current" in name:
        return "current"

    if "fair_housing" in name:
        return "washington_law"

    if "agency" in name:
        return "education"

    if "advanced_real_estate" in name:
        return "education"

    if "building_blocks" in name:
        return "education"

    if "brokerage" in name:
        return "education"

    if "home_prep" in name:
        return "education"

    return "general"


def detect_year(filename: str) -> int | None:

    match = re.search(
        r"(20\d{2})",
        filename,
    )

    if match:
        return int(match.group(1))

    return None


# ============================================================
# PDF extraction
# ============================================================

def clean_text(text: str) -> str:

    text = text.replace(
        "\x00",
        " ",
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n\s*\n+",
        "\n\n",
        text,
    )

    text = re.sub(
        r"(?<!\n)\n(?!\n)",
        " ",
        text,
    )

    return text.strip()


def extract_pdf_pages(
    pdf_path: Path,
) -> List[Tuple[int, str]]:

    pages = []

    try:

        reader = PdfReader(
            str(pdf_path)
        )

        for page_number, page in enumerate(
            reader.pages,
            start=1,
        ):

            try:
                text = page.extract_text() or ""

            except Exception as exc:

                print(
                    f"WARNING: "
                    f"{pdf_path.name} "
                    f"page {page_number}: "
                    f"{exc}"
                )

                text = ""

            text = clean_text(text)

            if text:
                pages.append(
                    (
                        page_number,
                        text,
                    )
                )

    except Exception as exc:

        print(
            f"ERROR reading "
            f"{pdf_path}: {exc}"
        )

    return pages


# ============================================================
# Chunking
# ============================================================

def chunk_text(
    text: str,
    chunk_size: int,
    overlap: int,
) -> List[str]:

    words = text.split()

    if not words:
        return []

    chunks = []

    start = 0

    while start < len(words):

        end = min(
            start + chunk_size,
            len(words),
        )

        chunk = " ".join(
            words[start:end]
        ).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        start = max(
            end - overlap,
            start + 1,
        )

    return chunks


# ============================================================
# Load PDFs
# ============================================================

def load_documents(
    pdf_dir: Path,
    chunk_size: int,
    overlap: int,
) -> List[Chunk]:

    chunks = []

    pdf_files = sorted(
        pdf_dir.glob("*.pdf")
    )

    if not pdf_files:

        print(
            f"ERROR: No PDFs found in "
            f"{pdf_dir}"
        )

        sys.exit(1)

    print()
    print("=" * 70)
    print("LOADING PDF DOCUMENTS")
    print("=" * 70)

    chunk_id = 0

    for pdf_path in pdf_files:

        document_type = classify_document(
            pdf_path.name
        )

        year = detect_year(
            pdf_path.name
        )

        pages = extract_pdf_pages(
            pdf_path
        )

        count = 0

        for page_number, page_text in pages:

            page_chunks = chunk_text(
                page_text,
                chunk_size,
                overlap,
            )

            for text in page_chunks:

                chunks.append(
                    Chunk(
                        text=text,
                        source=pdf_path.name,
                        page=page_number,
                        chunk_id=chunk_id,
                        document_type=document_type,
                        year=year,
                    )
                )

                chunk_id += 1
                count += 1

        print(
            f"  {pdf_path.name}"
        )

        print(
            f"      {count} chunks "
            f"[{document_type}]"
        )

    print()
    print(
        f"Total chunks: {len(chunks)}"
    )

    return chunks


# ============================================================
# Build embeddings
# ============================================================

def build_embeddings(
    chunks: List[Chunk],
    model: SentenceTransformer,
) -> np.ndarray:

    print()
    print("=" * 70)
    print("BUILDING EMBEDDINGS")
    print("=" * 70)

    texts = [
        chunk.text
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    embeddings = embeddings.astype(
        np.float32
    )

    print(
        f"Embedding dimension: "
        f"{embeddings.shape[1]}"
    )

    print(
        f"Vectors: "
        f"{embeddings.shape[0]}"
    )

    return embeddings


# ============================================================
# FAISS
# ============================================================

def build_faiss_index(
    embeddings: np.ndarray,
):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    print(
        f"FAISS vectors: "
        f"{index.ntotal}"
    )

    return index


# ============================================================
# Query analysis
# ============================================================

def analyze_query(
    query: str,
) -> Dict:

    q = query.lower()

    intent = "general"

    if any(
        x in q
        for x in [
            "submit an offer",
            "make an offer",
            "write an offer",
            "offer forms",
            "purchase offer",
            "forms do i need",
        ]
    ):
        intent = "offer"

    elif any(
        x in q
        for x in [
            "inspection",
            "inspect",
        ]
    ):
        intent = "inspection"

    elif any(
        x in q
        for x in [
            "disclosure",
            "disclosures",
        ]
    ):
        intent = "disclosure"

    elif any(
        x in q
        for x in [
            "agency",
            "buyer agency",
            "seller agency",
        ]
    ):
        intent = "agency"

    elif any(
        x in q
        for x in [
            "financing",
            "mortgage",
            "loan",
        ]
    ):
        intent = "financing"

    elif any(
        x in q
        for x in [
            "listing",
            "listing agreement",
        ]
    ):
        intent = "listing"

    elif any(
        x in q
        for x in [
            "fair housing",
            "discrimination",
        ]
    ):
        intent = "fair_housing"

    property_type = "unknown"

    if any(
        x in q
        for x in [
            "single family",
            "single-family",
            "house",
            "home",
        ]
    ):
        property_type = "single_family"

    elif any(
        x in q
        for x in [
            "condo",
            "condominium",
        ]
    ):
        property_type = "condo"

    elif any(
        x in q
        for x in [
            "townhouse",
            "townhome",
        ]
    ):
        property_type = "townhouse"

    elif "multifamily" in q:
        property_type = "multifamily"

    role = "unknown"

    if any(
        x in q
        for x in [
            "buyer",
            "my offer",
            "submit an offer",
            "i am buying",
        ]
    ):
        role = "buyer"

    elif any(
        x in q
        for x in [
            "seller",
            "my listing",
            "i am selling",
        ]
    ):
        role = "seller"

    return {
        "intent": intent,
        "property_type": property_type,
        "role": role,
    }


# ============================================================
# Lexical scoring
# ============================================================

def normalize_text(
    text: str,
) -> str:

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def tokenize(
    text: str,
) -> set:

    return set(
        normalize_text(text).split()
    )


def lexical_score(
    query: str,
    text: str,
) -> float:

    query_tokens = tokenize(
        query
    )

    text_tokens = tokenize(
        text
    )

    if not query_tokens:
        return 0.0

    matches = (
        query_tokens &
        text_tokens
    )

    return (
        len(matches)
        / len(query_tokens)
    )


# ============================================================
# Document priority
# ============================================================

def document_priority(
    chunk: Chunk,
    query_info: Dict,
) -> float:

    intent = query_info["intent"]

    if intent not in {
        "offer",
        "inspection",
        "disclosure",
        "agency",
        "financing",
        "listing",
        "fair_housing",
    }:
        return 0.0

    priorities = {
        "nwmls": 0.30,
        "washington_law": 0.25,
        "current": 0.20,
        "education": 0.05,
        "general": 0.00,
    }

    return priorities.get(
        chunk.document_type,
        0.0,
    )


# ============================================================
# Retrieval
# ============================================================

def retrieve(
    query: str,
    query_info: Dict,
    chunks: List[Chunk],
    embedding_model: SentenceTransformer,
    index,
    candidate_k: int,
) -> List[Dict]:

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(
        np.float32
    )

    search_k = min(
        max(candidate_k * 3, 30),
        len(chunks),
    )

    vector_scores, indices = index.search(
        query_embedding,
        search_k,
    )

    results = []

    for vector_score, idx in zip(
        vector_scores[0],
        indices[0],
    ):

        if idx < 0:
            continue

        chunk = chunks[idx]

        lexical = lexical_score(
            query,
            chunk.text,
        )

        priority = document_priority(
            chunk,
            query_info,
        )

        combined = (
            float(vector_score) * 0.65
            + lexical * 0.25
            + priority * 0.10
        )

        results.append(
            {
                "chunk": chunk,
                "vector_score": float(
                    vector_score
                ),
                "lexical_score": float(
                    lexical
                ),
                "priority": float(
                    priority
                ),
                "combined_score": float(
                    combined
                ),
            }
        )

    results.sort(
        key=lambda x:
            x["combined_score"],
        reverse=True,
    )

    return results[:candidate_k]


# ============================================================
# Reranking
# ============================================================

def rerank(
    query: str,
    candidates: List[Dict],
    reranker: CrossEncoder,
    final_k: int,
) -> List[Dict]:

    if not candidates:
        return []

    pairs = [
        (
            query,
            item["chunk"].text,
        )
        for item in candidates
    ]

    scores = reranker.predict(
        pairs,
        show_progress_bar=False,
    )

    for item, score in zip(
        candidates,
        scores,
    ):

        item["rerank_score"] = float(
            score
        )

    candidates.sort(
        key=lambda x:
            x["rerank_score"],
        reverse=True,
    )

    return candidates[:final_k]


# ============================================================
# Evidence check
# ============================================================

def sufficient_evidence(
    results: List[Dict],
) -> bool:

    if not results:
        return False

    best = results[0].get(
        "rerank_score",
        -999,
    )

    return best > -5.5


# ============================================================
# Context
# ============================================================

def build_context(
    results: List[Dict],
) -> str:

    parts = []

    for i, item in enumerate(
        results,
        start=1,
    ):

        chunk = item["chunk"]

        parts.append(
            f"""
SOURCE {i}
Document: {chunk.source}
Page: {chunk.page}
Type: {chunk.document_type}

{chunk.text}
""".strip()
        )

    return "\n\n".join(
        parts
    )


# ============================================================
# llama.cpp
# ============================================================

def check_llama(
    health_url: str,
) -> bool:

    try:

        response = requests.get(
            health_url,
            timeout=10,
        )

        if response.ok:

            print(
                "llama.cpp server: OK"
            )

            return True

        print(
            f"llama.cpp HTTP "
            f"{response.status_code}"
        )

    except requests.RequestException as exc:

        print(
            f"llama.cpp unavailable: "
            f"{exc}"
        )

    return False


def ask_llama(
    llama_url: str,
    question: str,
    query_info: Dict,
    context: str,
    max_tokens: int,
) -> str:

    system_prompt = """
You are a Washington State real-estate research assistant.

Use the supplied document evidence to answer the user's
question.

Rules:

- Do not invent forms.
- Do not invent legal requirements.
- Do not assume an old document represents the current
  NWMLS form set.
- Distinguish buyer documents from seller/listing documents.
- Distinguish required forms from optional addenda.
- Prefer NWMLS/current/legal sources over educational material.
- If the supplied evidence does not establish the answer,
  explicitly say so.
- Cite document name and page number.
- Do not use general model knowledge to fill missing evidence.

For forms questions, identify:
1. Form number/name
2. Purpose
3. Whether it appears required, optional, or conditional
4. Source and page
5. Any uncertainty or outdated-document warning
""".strip()

    user_prompt = f"""
Question:
{question}

Query analysis:
Intent: {query_info["intent"]}
Property type: {query_info["property_type"]}
Role: {query_info["role"]}

Retrieved documents:

{context}

Answer the question using the retrieved documents.
""".strip()

    payload = {
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "temperature": 0.1,
        "max_tokens": max_tokens,
        "stream": False,
    }

    try:

        response = requests.post(
            llama_url,
            json=payload,
            timeout=300,
        )

        response.raise_for_status()

        data = response.json()

        return (
            data["choices"][0]
            ["message"]
            ["content"]
            .strip()
        )

    except requests.RequestException as exc:

        print(
            f"ERROR calling llama.cpp: "
            f"{exc}"
        )

        return ""


# ============================================================
# Display sources
# ============================================================

def show_sources(
    results: List[Dict],
):

    print()
    print(
        "-" * 70
    )
    print("SOURCES")
    print(
        "-" * 70
    )

    seen = set()

    for item in results:

        chunk = item["chunk"]

        key = (
            chunk.source,
            chunk.page,
        )

        if key in seen:
            continue

        seen.add(key)

        print(
            f"- {chunk.source}, "
            f"page {chunk.page}"
        )


# ============================================================
# Display retrieval
# ============================================================

def show_retrieval(
    results: List[Dict],
):

    print()
    print(
        "-" * 70
    )
    print("RETRIEVED EVIDENCE")
    print(
        "-" * 70
    )

    for i, item in enumerate(
        results,
        start=1,
    ):

        chunk = item["chunk"]

        print(
            f"\n[{i}] "
            f"{chunk.source} "
            f"page {chunk.page}"
        )

        print(
            f"    vector:  "
            f"{item['vector_score']:.4f}"
        )

        print(
            f"    lexical: "
            f"{item['lexical_score']:.4f}"
        )

        print(
            f"    combined: "
            f"{item['combined_score']:.4f}"
        )

        print(
            f"    rerank:  "
            f"{item.get('rerank_score', 0):.4f}"
        )


# ============================================================
# Build RAG system
# ============================================================

def initialize_rag(
    args,
):

    print()
    print("=" * 70)
    print("INITIALIZING REAL ESTATE RAG")
    print("=" * 70)

    # --------------------------------------------------------
    # PDFs
    # --------------------------------------------------------

    chunks = load_documents(
        Path(args.pdf_dir),
        args.chunk_size,
        args.overlap,
    )

    # --------------------------------------------------------
    # Embedding model
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )
    print("LOADING EMBEDDING MODEL")
    print(
        "=" * 70
    )

    print(
        EMBEDDING_MODEL
    )

    embedding_model = (
        SentenceTransformer(
            EMBEDDING_MODEL
        )
    )

    # --------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------

    embeddings = build_embeddings(
        chunks,
        embedding_model,
    )

    # --------------------------------------------------------
    # FAISS
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )
    print("BUILDING FAISS INDEX")
    print(
        "=" * 70
    )

    index = build_faiss_index(
        embeddings
    )

    # --------------------------------------------------------
    # Reranker
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )
    print("LOADING CROSSENCODER")
    print(
        "=" * 70
    )

    print(
        RERANKER_MODEL
    )

    reranker = CrossEncoder(
        RERANKER_MODEL
    )

    # --------------------------------------------------------
    # llama.cpp
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )
    print("CHECKING LLAMA.CPP")
    print(
        "=" * 70
    )

    if not check_llama(
        args.health_url
    ):

        print()
        print(
            "Start llama.cpp before running "
            "the RAG assistant."
        )

        print()
        print(
            "Example:"
        )

        print(
            "./llama-server "
            "-m ~/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf "
            "--host 127.0.0.1 "
            "--port 8080"
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Done
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )
    print("RAG READY")
    print(
        "=" * 70
    )

    print(
        f"Documents/chunks: {len(chunks)}"
    )

    print(
        f"Candidate K:      {args.candidate_k}"
    )

    print(
        f"Final K:           {args.top_k}"
    )

    print()

    return (
        chunks,
        embedding_model,
        index,
        reranker,
    )


# ============================================================
# Interactive question
# ============================================================

def answer_question(
    question: str,
    args,
    chunks,
    embedding_model,
    index,
    reranker,
):

    query_info = analyze_query(
        question
    )

    print()
    print(
        f"Intent: {query_info['intent']} | "
        f"Property: {query_info['property_type']} | "
        f"Role: {query_info['role']}"
    )

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    candidates = retrieve(
        query=question,
        query_info=query_info,
        chunks=chunks,
        embedding_model=embedding_model,
        index=index,
        candidate_k=args.candidate_k,
    )

    # --------------------------------------------------------
    # Rerank
    # --------------------------------------------------------

    results = rerank(
        question,
        candidates,
        reranker,
        args.top_k,
    )

    if not sufficient_evidence(
        results
    ):

        print()
        print(
            "Not enough evidence was retrieved "
            "to answer this question safely."
        )

        show_retrieval(
            results
        )

        return

    # --------------------------------------------------------
    # Show retrieval
    # --------------------------------------------------------

    show_retrieval(
        results
    )

    # --------------------------------------------------------
    # Context
    # --------------------------------------------------------

    context = build_context(
        results
    )

    # --------------------------------------------------------
    # Qwen
    # --------------------------------------------------------

    print()
    print(
        "Asking local Qwen..."
    )

    answer = ask_llama(
        llama_url=args.llama_url,
        question=question,
        query_info=query_info,
        context=context,
        max_tokens=args.max_tokens,
    )

    if not answer:

        print(
            "No answer returned."
        )

        return

    # --------------------------------------------------------
    # Answer
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )
    print("ANSWER")
    print(
        "=" * 70
    )

    print()
    print(answer)

    show_sources(
        results
    )


# ============================================================
# Interactive shell
# ============================================================

def interactive_loop(
    args,
    chunks,
    embedding_model,
    index,
    reranker,
):

    print()
    print("=" * 70)
    print("REAL ESTATE AI ASSISTANT")
    print("=" * 70)

    print()
    print(
        "PDF knowledge base loaded."
    )

    print(
        "Ask questions about Washington real estate."
    )

    print()
    print(
        "Commands:"
    )

    print(
        "  /quit       Exit"
    )

    print(
        "  /exit       Exit"
    )

    print(
        "  /sources    Show loaded document count"
    )

    print()

    while True:

        try:

            question = input(
                "Real Estate AI > "
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError,
        ):

            print()
            print(
                "Goodbye."
            )

            break

        if not question:
            continue

        command = question.lower()

        if command in {
            "/quit",
            "/exit",
            "quit",
            "exit",
        }:

            print(
                "Goodbye."
            )

            break

        if command == "/sources":

            print()
            print(
                f"Loaded {len(chunks)} chunks "
                f"from the PDF knowledge base."
            )

            continue

        answer_question(
            question,
            args,
            chunks,
            embedding_model,
            index,
            reranker,
        )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Interactive Washington Real Estate "
            "RAG using local llama.cpp"
        )
    )

    parser.add_argument(
        "--pdf-dir",
        default=DEFAULT_PDF_DIR,
    )

    parser.add_argument(
        "--llama-url",
        default=DEFAULT_LLAMA_URL,
    )

    parser.add_argument(
        "--health-url",
        default=DEFAULT_HEALTH_URL,
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP,
    )

    parser.add_argument(
        "--candidate-k",
        type=int,
        default=DEFAULT_CANDIDATE_K,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_FINAL_K,
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # LOAD EVERYTHING BEFORE PROMPT
    # --------------------------------------------------------

    (
        chunks,
        embedding_model,
        index,
        reranker,
    ) = initialize_rag(
        args
    )

    # --------------------------------------------------------
    # NOW START PROMPT
    # --------------------------------------------------------

    interactive_loop(
        args,
        chunks,
        embedding_model,
        index,
        reranker,
    )


if __name__ == "__main__":
    main()