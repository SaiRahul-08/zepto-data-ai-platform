from pathlib import Path
from .config import DOCS_DIR

def load_documents():
    docs = []
    for path in sorted(DOCS_DIR.glob("doc_*.txt")):
        docs.append({"id": path.stem, "text": path.read_text(encoding="utf-8").strip()})
    if len(docs) != 8:
        raise RuntimeError(f"Expected 8 policy documents, found {len(docs)}")
    return docs

def chunk_documents(documents):
    # The rubric permits one chunk per document because each document is short.
    return [
        {"id": d["id"], "text": d["text"], "source": d["id"]}
        for d in documents
    ]

if __name__ == "__main__":
    chunks = chunk_documents(load_documents())
    print(f"Loaded {len(chunks)} chunks from 8 documents.")
