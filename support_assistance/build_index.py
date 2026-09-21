from src.vector_store import build_index
c = build_index()
print(f"Indexed {c.count()} chunks in ChromaDB.")
